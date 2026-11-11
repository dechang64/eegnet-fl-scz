"""N100 suppression, measured the way the dataset's own authors measure it.

`n100_classic_check.py` was written to answer "is this copy genuine", and it
picks a peak in 50-180 ms without touching the button-only condition. That is
NOT the paradigm's canonical metric, and the difference turns out to decide the
result (see AUDIT.md, Addendum 3):

    80-100 ms mean, self minus button-only, suppression = passive - corrected
        HC -2.21 uV   SZ -0.88 uV   diff -1.33   p = 0.0056   d = -0.61
    80-100 ms mean, no button-only subtraction
        HC -2.02 uV   SZ -1.48 uV   diff -0.54   p = 0.25     d = -0.26
    peak in 50-180 ms, no subtraction
        HC -1.81 uV   SZ -1.64 uV   diff -0.17   p = 0.72     d = -0.08

The first row reproduces Roach's own replication notebook (HC 2.2 uV,
SZ 0.88 uV, non-overlapping CIs). The recipe is theirs, transcribed:

  * component = mean amplitude over 80-100 ms, averaged over Fz, FCz, Cz
  * the button+tone epoch has the button-only epoch subtracted from it, to
    remove motor potentials that are time-locked to the press
  * suppression = passive_playback - corrected_button_tone
    (negative, because the N100 is a negative deflection)

Caveat carried over from the same notebook: the subtraction can manufacture a
positive N1 in subjects whose button-only condition is itself negative in this
window (subjects 60, 61, 64 are the named cases). The author flags this himself
- "we should revisit the practice of subtracting this control condition later".
Report the metric, keep the caveat attached.

Usage:
  python scripts/n100_canonical.py [--data data] [--out n100_canonical_check.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CHANS = ("Fz", "FCz", "Cz")
WIN = (0.080, 0.100)  # seconds after the event, inclusive


def window_amp(data_dir, cond, sid, fs, onset):
    """Mean amplitude in WIN, averaged over the three midline sites."""
    df = pd.read_parquet(Path(data_dir) / cond / f"{cond}_{sid}.parquet")
    t = np.arange(df.shape[0]) / fs - onset
    m = (t >= WIN[0] - 1e-9) & (t <= WIN[1] + 1e-9)
    if not m.any():
        raise ValueError(f"no samples inside {WIN} for {cond}_{sid}")
    return float(df.loc[m, list(CHANS)].to_numpy().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="n100_canonical_check.json")
    a = p.parse_args()

    meta = json.loads((Path(a.data) / "dataset_meta.json").read_text(encoding="utf-8"))
    fs, onset = meta["fs"], meta["event_onset_s"]
    labels = pd.read_csv(Path(a.data) / "labels.csv")
    lab = dict(zip(labels.subject_id.astype(str).str.replace(r"\.0$", "", regex=True),
                   labels.label))

    rows = []
    for sid in lab:
        self_ = window_amp(a.data, "selfsz", sid, fs, onset)
        passive = window_amp(a.data, "passivesz", sid, fs, onset)
        button = window_amp(a.data, "buttonsz", sid, fs, onset)
        corrected = self_ - button
        rows.append({
            "sid": sid,
            "label": int(lab[sid]),
            "self": self_,
            "self_minus_button": corrected,
            "passive": passive,
            "suppression": passive - corrected,
            # the three pipelines side by side, so the audit claim is checkable
            "suppression_no_subtraction": passive - self_,
        })
    d = pd.DataFrame(rows)

    hc = d[d.label == 0].suppression
    sz = d[d.label == 1].suppression
    t, pval = stats.ttest_ind(hc, sz, equal_var=False)
    sp = np.sqrt(((len(hc) - 1) * hc.var(ddof=1) + (len(sz) - 1) * sz.var(ddof=1))
                 / (len(hc) + len(sz) - 2))
    diff = hc.mean() - sz.mean()
    cohen_d = diff / sp
    # AUC a Gaussian shift of this size would produce - the ceiling a classifier
    # could reach if it only had this one feature, measured without error.
    auc_equiv = float(stats.norm.cdf(abs(cohen_d) / np.sqrt(2)))

    print(f"canonical metric: mean {WIN[0]*1000:.0f}-{WIN[1]*1000:.0f} ms "
          f"@ {'/'.join(CHANS)}, self-minus-button subtraction")
    print(f"  controls (HC, n={len(hc)}): {hc.mean():+.2f} uV "
          f"(SE {hc.sem():.2f}), 95% CI [{stats.t.interval(0.95, len(hc)-1, hc.mean(), hc.sem())[0]:+.2f}, "
          f"{stats.t.interval(0.95, len(hc)-1, hc.mean(), hc.sem())[1]:+.2f}]")
    print(f"  patients (SZ, n={len(sz)}): {sz.mean():+.2f} uV "
          f"(SE {sz.sem():.2f}), 95% CI [{stats.t.interval(0.95, len(sz)-1, sz.mean(), sz.sem())[0]:+.2f}, "
          f"{stats.t.interval(0.95, len(sz)-1, sz.mean(), sz.sem())[1]:+.2f}]")
    print(f"  HC - SZ = {diff:+.2f} uV   Welch t={t:.2f}  p={pval:.4f}  "
          f"Cohen d={cohen_d:+.2f}  AUC-equivalent={auc_equiv:.3f}")
    print(f"  (same data, no button-only subtraction: "
          f"{(d[d.label==0].suppression_no_subtraction.mean() - d[d.label==1].suppression_no_subtraction.mean()):+.2f} uV)")

    out = {
        "metric": {"window_ms": [WIN[0] * 1000, WIN[1] * 1000],
                   "channels": list(CHANS),
                   "correction": "selfsz - buttonsz",
                   "suppression": "passivesz - (selfsz - buttonsz)"},
        "hc": {"n": int(len(hc)), "mean": float(hc.mean()), "sd": float(hc.std()),
               "se": float(hc.sem())},
        "sz": {"n": int(len(sz)), "mean": float(sz.mean()), "sd": float(sz.std()),
               "se": float(sz.sem())},
        "diff": float(diff), "welch_t": float(t), "p": float(pval),
        "cohen_d": float(cohen_d), "auc_equivalent": auc_equiv,
        "diff_no_subtraction": float(d[d.label == 0].suppression_no_subtraction.mean()
                                     - d[d.label == 1].suppression_no_subtraction.mean()),
        "source": "recipe from broach/replicating-did-i-do-that-paper-analyses-with-r",
    }
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    d.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    print(f"wrote {a.out} and {Path(a.out).with_suffix('.csv')}")


if __name__ == "__main__":
    main()
