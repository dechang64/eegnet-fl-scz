"""Classical N100 suppression contrast (no ML) - the Ford 2014 effect itself.

Purpose: prove the data copy is genuine before trusting any number from it.
Two conditions must hold for the copy to be credible:
  1. the button-only condition shows no auditory evoked response after the
     event (it cannot - there is no tone);
  2. self-generated tones elicit a smaller (less negative) N100 than passive
     playback, in BOTH groups.

Metric note (audit discipline): a fixed 80-150 ms mean window mixes N100 with
P200 once per-subject latency jitter is present (subject 1: N100 trough at
~80 ms, P200 at ~130 ms), so the amplitude cancels and the metric reads near
zero. Peak amplitude (min within 50-180 ms) at Fz/FCz/Cz is used instead.

2026-11-10 result on the real data (81 subjects, third-party Zenodo copy):
  controls  n=32: suppression +1.80 +/- 1.99 uV (direction correct)
  patients  n=49: suppression +1.63 +/- 2.15 uV (direction correct)
  HC - SCZ difference +0.17 uV, permutation p(two-sided)=0.61 - NOT significant
  with this crude metric. Ford 2014's significant group contrast used a
  latency window around each subject's peak plus covariates; treat the group
  contrast here as a provenance check, not a replication claim.

Usage: python scripts/n100_classic_check.py [--data data --out n100_suppression_check.csv]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def n100_peak(data_dir, sid, cond, fs, onset, chans=("Fz", "FCz", "Cz")):
    vals = []
    for chan in chans:
        df = pd.read_parquet(Path(data_dir) / cond / f"{cond}_{sid}.parquet")
        t = np.arange(df.shape[0]) / fs - onset
        m = (t >= 0.050) & (t <= 0.180)
        vals.append(float(df.loc[m, chan].min()))
    return float(np.mean(vals))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="n100_suppression_check.csv")
    a = p.parse_args()

    meta = json.loads((Path(a.data) / "dataset_meta.json").read_text(encoding="utf-8"))
    fs, onset = meta["fs"], meta["event_onset_s"]
    labels = pd.read_csv(Path(a.data) / "labels.csv")
    lab = dict(zip(labels.subject_id.astype(str), labels.label))

    rows = []
    for sid in lab:
        s = n100_peak(a.data, sid, "selfsz", fs, onset)
        pv = n100_peak(a.data, sid, "passivesz", fs, onset)
        rows.append({"sid": sid, "label": lab[sid],
                     "n100_self": s, "n100_passive": pv,
                     "suppression": s - pv})
    d = pd.DataFrame(rows)
    hc, sz = d[d.label == 0], d[d.label == 1]
    print(f"controls (n={len(hc)}): suppression {hc.suppression.mean():+.2f} "
          f"+/- {hc.suppression.std():.2f} uV")
    print(f"patients (n={len(sz)}): suppression {sz.suppression.mean():+.2f} "
          f"+/- {sz.suppression.std():.2f} uV")
    diff = hc.suppression.mean() - sz.suppression.mean()
    rng = np.random.default_rng(0)
    vals, labs = d.suppression.to_numpy(), d.label.to_numpy()
    perm = np.array([rng.permutation(vals)[labs == 1].mean()
                     - rng.permutation(vals)[labs == 0].mean()
                     for _ in range(10000)])
    print(f"HC - SCZ: {diff:+.2f} uV, p(two-sided)="
          f"{float((np.abs(perm) >= abs(diff)).mean()):.4f}")
    d.to_csv(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
