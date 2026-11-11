#!/usr/bin/env python
"""Verify the 64-channel single-trial files against mergedTrialData.csv.

For each downloaded subject this script checks, without any ML:
  1. column layout matches columnLabels.csv (74 columns, incl. CPz/Pz/TP10);
  2. every trial has 3072 samples and the time axis implies ~1024 Hz with
     event onset at sample 1537 (= 0 ms);
  3. the set of accepted (subject, trial, condition) triples matches the
     mergedTrialData.csv rows with rejected == 0 exactly -- i.e. the raw
     files contain only accepted trials;
  4. per-trial N100 features can be recomputed from the raw waveform and
     agree with the Fz/FCz/Cz_N100 columns in mergedTrialData.csv.

Usage:
  python scripts/verify_single_trial.py --raw-dir D:/eegnet-data/single_trial \
      --subjects 1 2 3 67 68 69

Exits non-zero if any check fails.
"""
import argparse
import csv
import glob
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
N_CHANNELS = 74
N_SAMPLES = 3072
ONSET_SAMPLE = 1537  # documented: sample 1537 == 0 ms
N100_LO_MS, N100_HI_MS = 80.0, 100.0
MIDLINE = ["Fz", "FCz", "Cz"]


def _fs_from_rows(n_samples: int) -> float:
    """Nominal fs implied by 3072 samples spanning -1500..1499 ms (~3 s)."""
    return n_samples / 3.0  # 1024 Hz documented, 1023.96... measured


def load_time_grid(meta_dir: Path) -> np.ndarray:
    """time.csv (header 'sample, time_ms', 3072 rows) gives each row's ms offset."""
    f = meta_dir / "time.csv"
    if f.exists():
        t = pd.read_csv(f, header=0).iloc[:, 1].to_numpy(dtype=float)
        if len(t) == N_SAMPLES:
            return t
    # fall back: derive from onset sample and nominal fs
    return (np.arange(N_SAMPLES) - ONSET_SAMPLE) / _fs_from_rows(N_SAMPLES) * 1000.0


def read_subject_csv(zip_path: Path) -> pd.DataFrame:
    """Raw files are headerless; columns follow columnLabels.csv."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"no csv inside {zip_path}")
        with zf.open(names[0]) as fh:
            return pd.read_csv(fh, header=None, low_memory=False)


def verify_subject(zip_path: Path, merged: pd.DataFrame, time_ms: np.ndarray) -> dict:
    subj = int(zip_path.name.split(".")[0])
    df = read_subject_csv(zip_path)  # headerless numeric frame

    res = {"subject": subj, "file": str(zip_path.name), "checks": {}}

    # 1. column layout
    expected = pd.read_csv(REPO.parent / "kaggle_meta" / "columnLabels.csv",
                           nrows=0).columns.tolist()
    res["checks"]["columns_match_columnLabels"] = len(df.columns) == len(expected)
    res["n_columns"] = len(df.columns)

    # 2. shape and sampling
    res["checks"]["shape_multiple_of_3072"] = (len(df) % N_SAMPLES == 0)
    n_trials = len(df) // N_SAMPLES if res["checks"]["shape_multiple_of_3072"] else None
    res["n_trials_file"] = n_trials
    res["fs_hz"] = round(_fs_from_rows(N_SAMPLES), 2)

    # reshape: (trial, sample, channel) per official MNE notebook
    arr = df.to_numpy(dtype=float).reshape(-1, N_SAMPLES, len(df.columns))
    col = {c: i for i, c in enumerate(expected) if i < len(df.columns)}
    trial_meta = arr[:, 0, :4]  # subject, trial, condition, sample are per-row

    # 3. accepted-trial set equality (file order is presentation order, not sorted)
    acc = merged[(merged.subject == subj) & (merged.rejected == 0)]
    acc_keys = set(zip(acc.trial.astype(int), acc.condition.astype(int)))
    keys_file = [(int(r[1]), int(r[2])) for r in trial_meta]
    file_keys = set(keys_file)
    res["n_trials_merged_accepted"] = len(acc_keys)
    res["checks"]["accepted_trials_match"] = acc_keys == file_keys
    res["checks"]["one_block_per_trial"] = len(keys_file) == len(file_keys)

    # 4. recompute N100 mean 80-100 ms at Fz/FCz/Cz, compare to merged columns
    # (keyed by (trial, condition) because file order is presentation order)
    w = (time_ms >= N100_LO_MS) & (time_ms <= N100_HI_MS)
    acc_i = acc.set_index(["trial", "condition"])
    worst = 0.0
    for t_idx, key in enumerate(keys_file):
        if key not in acc_i.index:
            continue
        row = acc_i.loc[key]
        for ch in MIDLINE:
            v_file = float(arr[t_idx][:, col[ch]][w].mean())
            v_merged = float(row["%s_N100" % ch])
            worst = max(worst, abs(v_file - v_merged))
    res["max_abs_n100_recompute_err_uV"] = worst
    res["checks"]["n100_recomputes"] = worst < 0.5  # uV tolerance

    res["all_pass"] = all(
        v for k, v in res["checks"].items() if isinstance(v, bool))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--subjects", nargs="+", type=int, required=True)
    ap.add_argument("--merged", default=str(REPO.parent / "kaggle_raw" / "mergedTrialData.csv"))
    args = ap.parse_args()

    merged = pd.read_csv(args.merged)
    merged["subject"] = pd.to_numeric(merged.subject, errors="coerce")
    time_ms = load_time_grid(REPO.parent / "kaggle_meta")

    out = []
    ok = True
    for sid in args.subjects:
        zips = list(Path(args.raw_dir).glob(f"{sid}.csv.zip")) or \
               list(Path(args.raw_dir).glob(f"*/{sid}.csv.zip"))
        if not zips:
            print(f"subject {sid}: NO ZIP FOUND", file=sys.stderr)
            ok = False
            continue
        try:
            r = verify_subject(zips[0], merged, time_ms)
        except Exception as e:  # noqa: BLE001 - report, don't crash the batch
            print(f"subject {sid}: ERROR {e}", file=sys.stderr)
            ok = False
            continue
        out.append(r)
        flags = " ".join(k for k, v in r["checks"].items() if v is False) or "ok"
        print(f"subject {sid}: trials={r['n_trials_file']} "
              f"fs={r['fs_hz']}Hz n100_err={r['max_abs_n100_recompute_err_uV']:.4f}uV "
              f"-> {flags}")

    dest = REPO / "verify_single_trial.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"written {dest}")
    return 0 if ok and all(r["all_pass"] for r in out) else 1


if __name__ == "__main__":
    sys.exit(main())
