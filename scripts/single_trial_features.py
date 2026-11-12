"""Single-trial N100 features, condition mapping, and a leak-free baseline.

Three questions, one pass over the raw archives:

1. **Which numeric code is which condition?**  The single-trial files carry a
   `condition` column with values 1/2/3 but no legend. `data/{selfsz,
   passivesz,buttonsz}/*.parquet` is the same data already split by condition,
   so the subject-level mean amplitude of each code can be matched against each
   parquet condition. A 3x3 correlation matrix over 81 subjects identifies the
   mapping empirically instead of guessing.

2. **Does the canonical N100 survive the trip through the raw files?**  Compare
   the per-subject canonical suppression computed here (80-100 ms mean @
   Fz/FCz/Cz, passive minus self-minus-button) with `n100_canonical_check.json`.

3. **Is one feature enough?**  Subject-level AUC of that single canonical feature
   is the ceiling a perfect measurement of it would give. A per-trial logistic
   regression on the same 80-100 ms amplitude, scored under subject-grouped
   10-fold CV, is what a decoder gets from single trials.

Outputs `single_trial_features.json` and caches per-trial amplitudes to
`D:/eegnet-data/single_trial_features.npz` so later analyses skip the 17 GB
re-read.

Usage:
  python scripts/single_trial_features.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CHANS = ("Fz", "FCz", "Cz")
WIN = (80.0, 100.0)  # ms after the event
N_SAMPLES = 3072
COND_NAMES = ("selfsz", "passivesz", "buttonsz")

WIN_IDX: np.ndarray | None = None
COLS: list[str] = []


def time_grid() -> np.ndarray:
    t = pd.read_csv(REPO.parent / "kaggle_meta" / "time.csv", header=0)
    return t.iloc[:, 1].to_numpy(dtype=float)


def load_trials(zip_path: Path, cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (amplitude n_trials, condition n_trials, trial id n_trials)."""
    with zipfile.ZipFile(zip_path) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(name) as fh:
            df = pd.read_csv(fh, header=None, names=cols, usecols=COLS,
                             low_memory=False)
    heads = df[df["sample"] == 1][["trial", "condition"]].reset_index(drop=True)
    amp = df[list(CHANS)].to_numpy(dtype=np.float32).reshape(-1, N_SAMPLES, len(CHANS))
    amp = amp[:, WIN_IDX, :].mean(axis=(1, 2))
    return amp, heads["condition"].to_numpy(), heads["trial"].to_numpy()


def parquet_means(data_dir: Path, sid: int, fs: float,
                  onset: float) -> dict[str, float]:
    out = {}
    lo, hi = WIN[0] / 1000.0, WIN[1] / 1000.0
    for cond in COND_NAMES:
        p = data_dir / cond / f"{cond}_{sid}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        t = np.arange(df.shape[0]) / fs - onset
        m = (t >= lo - 1e-9) & (t <= hi + 1e-9)
        out[cond] = float(df.loc[m, list(CHANS)].to_numpy().mean())
    return out


def main() -> int:
    global WIN_IDX, COLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="D:/eegnet-data/single_trial")
    ap.add_argument("--raw-dir2", default="D:/eegnet-data/single_trial_ds2")
    ap.add_argument("--data", default="data")
    ap.add_argument("--cache", default="D:/eegnet-data/single_trial_features.npz")
    ap.add_argument("--subjects", default="", help="debug: comma list only")
    args = ap.parse_args()

    cols = pd.read_csv(REPO.parent / "kaggle_meta" / "columnLabels.csv",
                       nrows=0).columns.tolist()
    COLS = cols
    grid = time_grid()
    WIN_IDX = np.where((grid >= WIN[0] - 0.1) & (grid <= WIN[1] + 0.1))[0]
    meta = json.loads((REPO / args.data / "dataset_meta.json").read_text(
        encoding="utf-8"))
    fs, onset = float(meta["fs"]), float(meta["event_onset_s"])
    print(f"window {WIN[0]:.0f}-{WIN[1]:.0f} ms -> {len(WIN_IDX)} samples",
          flush=True)
    labels = pd.read_csv(REPO / "data" / "labels.csv")
    lab = dict(zip(labels.subject_id.astype(int), labels.label))

    raw_dirs = [Path(args.raw_dir), Path(args.raw_dir2)]
    cache = Path(args.cache)

    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        sids = list(z["sids"])
        per_trial = {int(s): z[f"amp_{s}"] for s in sids}
        per_cond = {int(s): z[f"cond_{s}"] for s in sids}
        print(f"loaded cache {cache} ({len(sids)} subjects)", flush=True)
    else:
        per_trial, per_cond, sids = {}, {}, []
        t0 = time.time()
        todo = ([int(x) for x in args.subjects.split(",") if x.strip()]
                if args.subjects else sorted(lab))
        for sid in todo:
            zp = next((d / f"{sid}.csv.zip" for d in raw_dirs
                       if (d / f"{sid}.csv.zip").exists()), None)
            if zp is None:
                print(f"MISSING subject {sid}")
                return 1
            amp, cond, trial = load_trials(zp, cols)
            per_trial[sid], per_cond[sid] = amp, cond
            sids.append(sid)
            if sid % 10 == 1:
                print(f"  {sid:3d}: {len(amp):3d} trials, "
                      f"{time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(cache, sids=np.array(sids),
                            **{f"amp_{s}": per_trial[s] for s in sids},
                            **{f"cond_{s}": per_cond[s] for s in sids})
        print(f"cached to {cache}", flush=True)

    # ---- 1. condition code <-> parquet condition -------------------------
    data_dir = REPO / args.data
    hit = [s for s in sids if (data_dir / "selfsz" / f"selfsz_{s}.parquet").exists()]
    codes = sorted({int(c) for s in sids for c in np.unique(per_cond[s])})
    mat = pd.DataFrame(index=[f"code{c}" for c in codes], columns=list(COND_NAMES),
                       dtype=float)
    for cond in COND_NAMES:
        ref = np.array([parquet_means(data_dir, s, fs, onset).get(cond, np.nan)
                        for s in hit])
        for c in codes:
            got = np.array([per_trial[s][per_cond[s] == c].mean() for s in hit])
            r = stats.pearsonr(ref, got)[0] if np.isfinite(got).all() else np.nan
            mat.loc[f"code{c}", cond] = round(float(r), 3)
    print("\ncorrelation of single-trial code means with parquet conditions:")
    print(mat.to_string())
    code_of = {cond: int(mat[cond].astype(float).idxmax().replace("code", ""))
               for cond in COND_NAMES}
    print("resolved mapping:", code_of, flush=True)

    # ---- 2. canonical suppression from the raw files ---------------------
    rows = []
    for s in sids:
        a, c = per_trial[s], per_cond[s]
        self_ = a[c == code_of["selfsz"]].mean()
        passive = a[c == code_of["passivesz"]].mean()
        button = a[c == code_of["buttonsz"]].mean()
        rows.append({"sid": s, "label": int(lab[s]), "self": float(self_),
                     "passive": float(passive), "button": float(button),
                     "suppression": float(passive - (self_ - button)),
                     "n_self": int((c == code_of["selfsz"]).sum())})
    d = pd.DataFrame(rows)
    hc, sz = d[d.label == 0].suppression, d[d.label == 1].suppression
    t, p = stats.ttest_ind(hc, sz, equal_var=False)
    sp = np.sqrt(((len(hc) - 1) * hc.var(ddof=1) + (len(sz) - 1) * sz.var(ddof=1)) /
                 (len(hc) + len(sz) - 2))
    dd = (hc.mean() - sz.mean()) / sp
    auc_subj = float(stats.norm.cdf(abs(dd) / np.sqrt(2)))
    # rank AUC, no Gaussian assumption
    allv = np.r_[hc, sz]
    ranks = stats.rankdata(-allv)  # more negative suppression = more HC-like
    n1, n0 = len(hc), len(sz)
    auc_rank = float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    print(f"\ncanonical suppression from raw files: HC {hc.mean():+.2f} uV "
          f"(n={n1}), SZ {sz.mean():+.2f} uV (n={n0})")
    print(f"  HC-SZ {hc.mean()-sz.mean():+.2f} uV  Welch p={p:.4f}  d={dd:+.2f}  "
          f"AUC(gauss)={auc_subj:.3f}  AUC(rank)={auc_rank:.3f}")

    # ---- 3. trial-level baseline, subject-grouped -----------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    X = np.concatenate([per_trial[s][per_cond[s] == code_of["selfsz"]]
                        for s in sids])[:, None]
    y = np.concatenate([np.full((per_cond[s] == code_of["selfsz"]).sum(), lab[s])
                        for s in sids])
    g = np.concatenate([np.full((per_cond[s] == code_of["selfsz"]).sum(), s)
                        for s in sids])
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=20261111)
    oof = np.full(len(y), np.nan)
    for tr_i, te_i in skf.split(X, y):
        mu, sd = X[tr_i].mean(), X[tr_i].std() + 1e-9
        clf = LogisticRegression().fit((X[tr_i] - mu) / sd, y[tr_i])
        oof[te_i] = clf.predict_proba((X[te_i] - mu) / sd)[:, 1]
    trial_auc = float(roc_auc_score(y, oof))
    # subject decision = mean trial probability
    subj = pd.DataFrame({"g": g, "y": y, "p": oof}).groupby("g").agg(
        y=("y", "first"), p=("p", "mean"))
    subj_auc = float(roc_auc_score(subj.y, subj.p))
    print(f"trial-level logistic on 80-100 ms amplitude ({len(y)} trials, "
          f"subject-grouped 10-fold): AUC={trial_auc:.3f}; "
          f"subject decision AUC={subj_auc:.3f}")

    out = {
        "condition_code_map": code_of,
        "condition_correlation": mat.to_dict(),
        "canonical_from_raw": {
            "hc": {"n": int(n1), "mean": float(hc.mean())},
            "sz": {"n": int(n0), "mean": float(sz.mean())},
            "diff": float(hc.mean() - sz.mean()), "welch_p": float(p),
            "cohen_d": float(dd), "auc_gauss": auc_subj, "auc_rank": auc_rank,
        },
        "trial_level_baseline": {"n_trials": int(len(y)), "auc": trial_auc,
                                 "subject_decision_auc": subj_auc},
        "n_subjects": len(sids),
    }
    dest = REPO / "single_trial_features.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"written {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
