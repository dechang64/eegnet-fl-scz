"""Does removing trial averaging actually help? Measured, not assumed.

The averaged dataset (ERPdata.csv) gives one waveform per subject per
condition, so a subject-level classifier has n=81. mergedTrialData.csv holds
23201 usable single trials, which invites the hope that n becomes 23201. This
script quantifies what that really buys, and what it does not.

Three evaluation protocols, all on the same self-condition trials:

  A  single trial, random 5-fold, subject labels as targets
     -> WRONG. Trials from one subject land on both sides of the split, so the
        model can memorise the person instead of the diagnosis. This is the
        pseudo-replication already recorded as audit item B2.
  B  single trial, subject-grouped 5-fold, per-subject predictions averaged
     -> honest. Every trial of a held-out subject is predicted by a model that
        never saw that subject, then the trial probabilities are averaged into
        one score per subject.
  C  trials averaged into one feature vector per subject, then subject-grouped
     -> honest, and equivalent in spirit to using ERPdata.csv (only the 18
        component features are available here, not the full waveforms).

Why n does not become 23201: trials cluster within people. ICC(1) on Fz_N100
is ~0.04, so the effective sample size is
    n_eff = n_subj * k / (1 + (k - 1) * ICC)  ~ 1600,
not 23201. And the quantity a diagnostic claim needs - "will this work on a
person I have never seen" - is still answered by 81 people, because that is
how many independent people exist in the data.

What single trials DO buy is reliability of each subject's measurement.
Single-trial N100 reliability is r=0.04; averaging ~97 trials lifts it to
0.82-0.89 (Spearman-Brown). That is the real gain: less attenuation, so a
smaller true group difference becomes visible. It does not add people.

Run: python scripts/single_trial_check.py --trials _zenodo/mergedTrialData.csv \
         --demographics _zenodo/demographic.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def icc1(vals, groups):
    """ICC(1): share of total variance that sits between subjects."""
    df = pd.DataFrame({"v": vals, "g": groups})
    sse = ((df.v - df.groupby("g").v.transform("mean")) ** 2).sum()
    ssb = ((df.groupby("g").v.transform("mean") - df.v.mean()) ** 2).sum()
    k = df.groupby("g").size()
    n, g_n = len(df), len(k)
    msb, msw = ssb / (g_n - 1), sse / (n - g_n)
    k0 = (n - (k ** 2).sum() / n) / (g_n - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def split_half_reliability(df, col, seed=0):
    """Random split-half correlation, Spearman-Brown corrected to full length."""
    rng = np.random.default_rng(seed)
    a, b = [], []
    for _, g in df.groupby("subject"):
        v = g[col].to_numpy()
        idx = rng.permutation(len(v))
        h = len(v) // 2
        a.append(v[idx[:h]].mean())
        b.append(v[idx[h:]].mean())
    r = float(np.corrcoef(a, b)[0, 1])
    return 2 * r / (1 + r)


def hanley_se(auc, n_pos, n_neg):
    """Hanley-McNeil standard error of an AUC."""
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    return float(np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
                          + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)))


def min_detectable_auc(n_pos, n_neg, power_z=2.80):
    """Smallest AUC reaching 80% power (2.80 = 1.96 + 0.84), by bisection."""
    lo, hi = 0.5, 0.99
    for _ in range(80):
        mid = (lo + hi) / 2
        if (mid - 0.5) / hanley_se(mid, n_pos, n_neg) < power_z:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", default="_zenodo/mergedTrialData.csv")
    p.add_argument("--demographics", default="_zenodo/demographic.csv")
    p.add_argument("--condition", type=int, default=1,
                   help="1=self-generated tone (the condition we classify on)")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--out", default="single_trial_check.json")
    a = p.parse_args()

    d = pd.read_csv(a.trials, skipinitialspace=True)
    d = d[(d.rejected == 0) & (d.condition == a.condition)].copy()
    dem = pd.read_csv(a.demographics, skipinitialspace=True)
    lab = dict(zip(dem.iloc[:, 0].astype(str), dem["group"]))
    d["subject"] = d.subject.astype(str)
    if not set(d.subject) <= set(lab):
        raise SystemExit("trial file has subjects absent from the demographics "
                         "table - labels would be guessed")

    feats = [c for c in d.columns if c.endswith(("_N100", "_P200"))]
    if not feats:
        raise SystemExit(f"no component columns in {a.trials}")

    X = d[feats].to_numpy()
    y = d.subject.map(lab).to_numpy().astype(int)
    g = d.subject.to_numpy()
    n_subj = len(set(g))
    print(f"self condition: {len(d)} trials, {n_subj} subjects, "
          f"{len(feats)} features (N100/P200 amplitudes per site)")

    pipe = lambda: make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=3000,
                                                    class_weight="balanced"))
    acc = {"A_random_kfold": [], "B_grouped_trial": [], "C_subject_mean": []}
    sm = d.groupby("subject")[feats].mean()
    sy = np.array([lab[s] for s in sm.index])
    Xm, sg = sm.to_numpy(), np.arange(len(sy))

    for r in range(a.reps):
        pr = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            pr[te] = pipe().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        acc["A_random_kfold"].append(roc_auc_score(y, pr))

        per = {}
        for tr, te in GroupKFold(5).split(X, y, g):
            mdl = pipe().fit(X[tr], y[tr])
            for s, v in zip(g[te], mdl.predict_proba(X[te])[:, 1]):
                per.setdefault(s, []).append(v)
        sc = {s: np.mean(v) for s, v in per.items()}
        acc["B_grouped_trial"].append(
            roc_auc_score([lab[s] for s in sc], [sc[s] for s in sc]))

        p3 = np.zeros(len(sy))
        for tr, te in GroupKFold(5).split(Xm, sy, sg):
            p3[te] = pipe().fit(Xm[tr], sy[tr]).predict_proba(Xm[te])[:, 1]
        acc["C_subject_mean"].append(roc_auc_score(sy, p3))

    print("\nAUC by protocol (mean +/- sd over repeats):")
    labels = {"A_random_kfold": "A single trial, random 5-fold  (WRONG, leaked)",
              "B_grouped_trial": "B single trial, grouped + aggregated",
              "C_subject_mean": "C averaged per subject, grouped"}
    for k, v in acc.items():
        print(f"  {labels[k]:<46} {np.mean(v):.3f} +/- {np.std(v):.3f}")

    point = np.mean(acc["B_grouped_trial"])
    n_pos, n_neg = int((sy == 1).sum()), int((sy == 0).sum())
    se = hanley_se(point, n_pos, n_neg)
    lo, hi = point - 1.96 * se, point + 1.96 * se
    floor = min_detectable_auc(n_pos, n_neg)
    print(f"\nB point estimate {point:.3f}, 95% CI [{lo:.3f}, {hi:.3f}] "
          f"(n={n_pos} SCZ / {n_neg} control)")
    print(f"  CI excludes 0.5 ? {lo > 0.5}")
    print(f"  smallest AUC reaching 80% power at this n: {floor:.3f}")

    icc = icc1(d.Fz_N100.to_numpy(), g)
    k = len(d) / n_subj
    n_eff = n_subj * k / (1 + (k - 1) * icc)
    print(f"\nClustering: ICC(1) on Fz_N100 = {icc:.3f}, k = {k:.0f} trials/subject")
    print(f"  nominal n = {len(d)}  ->  effective n = {n_eff:.0f}"
          f"  ->  independent people = {n_subj}")
    print(f"Reliability of Fz_N100: single trial r={icc:.3f}, "
          f"averaged r={split_half_reliability(d, 'Fz_N100'):.3f}")

    Path(a.out).write_text(json.dumps({
        "condition": a.condition, "n_trials": len(d), "n_subjects": n_subj,
        "features": feats,
        "auc": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
                for k, v in acc.items()},
        "protocol_B_ci95": [float(lo), float(hi)],
        "ci_excludes_chance": bool(lo > 0.5),
        "min_detectable_auc_80power": float(floor),
        "icc_fz_n100": icc, "n_effective": float(n_eff),
        "reliability_avg_fz_n100": float(split_half_reliability(d, "Fz_N100")),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
