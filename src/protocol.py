"""Subject-based cross-validation and honest metric reporting.

AUDIT 2026-11-10
---------------
B2 (high): bootstrap CIs resampled *epochs*, not *subjects*. With ~10 epochs per
    subject that is pseudo-replication: the CI is far too narrow and any
    "significant" claim built on it is unsupported. `report_metrics` now does a
    cluster bootstrap over subjects whenever `groups` is supplied, and refuses
    to pretend otherwise when it is not.
B3 (high): a subject appearing with two different labels is a data-integrity
    fault that silently poisons group CV. It now raises.
B5 (med): `confusion_matrix(...).ravel()` unpacked into 4 values crashes when
    only one class is present; `roc_auc_score` raises likewise. Both are now
    guarded and return NaN with an explicit warning instead of crashing a
    5-fold run at fold 4.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

NAN = float("nan")


class SubjectBasedCV:
    """Grouped, stratified K-fold: no subject may appear in train and test."""

    def __init__(self, n_splits: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed
        self.cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                       random_state=seed)

    def split(self, subjects: np.ndarray, labels: np.ndarray):
        subjects, labels = np.asarray(subjects), np.asarray(labels)
        if len(subjects) != len(labels):
            raise ValueError(
                f"subjects/labels length mismatch: {len(subjects)} vs "
                f"{len(labels)}")
        uniq = np.unique(subjects)
        if len(uniq) < self.n_splits:
            raise ValueError(
                f"only {len(uniq)} subjects for {self.n_splits}-fold CV")
        # B3: one label per subject, or the grouping is meaningless
        for s in uniq:
            labs = set(np.unique(labels[subjects == s]).tolist())
            if len(labs) > 1:
                raise RuntimeError(
                    f"subject {s!r} carries conflicting labels {sorted(labs)} "
                    "- fix the metadata before any CV")
        for tr, te in self.cv.split(np.zeros(len(subjects)), labels,
                                    groups=subjects):
            overlap = set(subjects[tr]) & set(subjects[te])
            if overlap:
                raise RuntimeError(
                    "LEAKAGE DETECTED: subject(s) "
                    f"{sorted(overlap)} in both folds")
            yield tr, te


def _safe_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return NAN
    return float(roc_auc_score(y_true, y_score))


def report_metrics(y_true, y_pred, y_score, groups=None, n_boot=1000,
                   seed=42):
    """Point metrics + a subject-cluster bootstrap CI for AUC.

    `groups` must be the subject id of each epoch. Without it the CI falls back
    to an epoch-level bootstrap, which is anti-conservative; a warning is
    emitted so no downstream claim can quietly inherit the inflated precision.
    """
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    single_class = len(np.unique(y_true)) < 2
    if single_class:
        warnings.warn("only one class in y_true: AUC/CI undefined (NaN)",
                      RuntimeWarning, stacklevel=2)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred,
                                      labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) else NAN
    spec = float(tn / (tn + fp)) if (tn + fp) else NAN
    auc = _safe_auc(y_true, y_score)

    rng = np.random.default_rng(seed)
    boots, n_units = [], 0
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y_true):
            raise ValueError("groups/y_true length mismatch")
        uniq = np.unique(groups)
        n_units = len(uniq)                      # B2: resample SUBJECTS
        index_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
        for _ in range(n_boot):
            pick = rng.integers(0, n_units, n_units)
            idx = np.concatenate([index_by_group[uniq[i]] for i in pick])
            val = _safe_auc(y_true[idx], y_score[idx])
            if not np.isnan(val):
                boots.append(val)
    else:
        warnings.warn(
            "report_metrics called without `groups`: the AUC CI resamples "
            "epochs and is therefore anti-conservative (pseudo-replication). "
            "Pass subject ids.", RuntimeWarning, stacklevel=2)
        n_units = len(y_true)
        for _ in range(n_boot):
            idx = rng.integers(0, n_units, n_units)
            val = _safe_auc(y_true[idx], y_score[idx])
            if not np.isnan(val):
                boots.append(val)

    ci = ((float(np.percentile(boots, 2.5)),
           float(np.percentile(boots, 97.5))) if boots else (NAN, NAN))
    return dict(acc=acc, auc=auc, sens=sens, spec=spec, f1=f1,
                auc_ci=ci, boot_units=int(n_units),
                n_epochs=int(len(y_true)), single_class=bool(single_class))


def aggregate_folds(folds, n_subjects_per_fold):
    """Mean/sd across folds plus a subject-weighted pooled AUC (mean of folds
    over-weights small folds; both numbers are reported, none is hidden)."""
    keys = ("acc", "auc", "sens", "spec", "f1")
    agg = {}
    for k in keys:
        vals = np.array([f[k] for f in folds], dtype=float)
        agg[k] = [float(np.nanmean(vals)), float(np.nanstd(vals))]
    weights = np.asarray(n_subjects_per_fold, dtype=float)
    aucs = np.array([f["auc"] for f in folds], dtype=float)
    ok = ~np.isnan(aucs)
    agg["auc_subject_weighted"] = [
        float(np.sum(aucs[ok] * weights[ok]) / np.sum(weights[ok]))
        if ok.any() else NAN, 0.0]
    return agg
