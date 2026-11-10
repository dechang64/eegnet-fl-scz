"""Turn the raw Kaggle export into the layout src/data.py expects.

Why this module exists
----------------------
`src/data.py` deliberately refuses to guess: it wants
`data/<condition>/<condition>_<subject>.parquet`, each file
`(channels, time)` with channels in `CHANNELS` order. The Kaggle download is
not that. Kaggle ships `ERPdata.csv` (long: one averaged ERP waveform per
subject x condition x electrode), `demographic.csv` (subject metadata) and
`mergedTrialData.csv` (summary features, not waveforms). Something has to
translate, and that something must be explicit and checkable - otherwise the
translation itself becomes an unexamined assumption, which is the failure mode
the whole audit was about.

Nothing here assumes a column name. Roles are detected from aliases; if a role
cannot be resolved uniquely the function raises and prints the columns it saw,
so the caller can pass the real names on the command line.

Facts this module encodes (from the dataset's own description, verified
2026-11-10):
  - 81 subjects: 49 patients with schizophrenia, 32 controls
  - 3 conditions: button->tone (self), passive tone, button alone
  - single trials are 3 s, -1.5 s to +1.5 s around the event
  - 9 analysis sites: Fz FCz Cz FC3 FC4 C3 C4 CP3 CP4
  - single-trial 64-channel waveforms are public for only ~24 subjects
Sites and conditions are hard requirements; sampling rate is derived from the
time column rather than assumed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CHANNELS, CONDITIONS

# Kaggle's condition tokens are not documented per export, so accept the
# plausible spellings and let the caller override with --condition-map.
CONDITION_TOKENS = {
    "selfsz": {"self", "selfsz", "1", "c1", "condition1", "cond1", "active",
               "buttontone", "self_tone", "selfgenerated"},
    "passivesz": {"passive", "passivesz", "2", "c2", "condition2", "cond2",
                  "listen", "passivetone", "passive_tone", "playback"},
    "buttonsz": {"button", "buttonsz", "3", "c3", "condition3", "cond3",
                 "buttononly", "button_only", "pressonly", "press"},
}

COLUMN_ALIASES = {
    "subject": ["subject", "subjectid", "subj", "sub", "sid", "participant",
                "participantid", "subjid"],
    "condition": ["condition", "cond", "task", "event", "eventtype", "stim"],
    "electrode": ["electrode", "electrodename", "channel", "channelname",
                  "site", "ch"],
    "time": ["time", "timems", "t", "latency", "ms", "sample", "timepoint"],
    "value": ["value", "amplitude", "erp", "voltage", "mean", "uv",
              "amplitudeuv", "signal", "potential"],
    "label": ["diagnosis", "dx", "group", "label", "class", "status",
              "subjecttype", "category"],
}

SCZ_TOKENS = {"1", "sz", "scz", "patient", "patients", "schizophrenia",
              "case", "ill", "affected"}


class IngestError(ValueError):
    """Raised instead of emitting a plausible-looking wrong dataset."""


def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def detect_column(df: pd.DataFrame, role: str) -> str | None:
    """Resolve one role to exactly one column name, or None. Never guesses."""
    wanted = COLUMN_ALIASES[role]
    by_norm = {_norm(c): c for c in df.columns}
    for w in wanted:                       # exact alias wins, in priority order
        if w in by_norm:
            return by_norm[w]
    hits = [by_norm[n] for n in by_norm if any(w in n for w in wanted)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise IngestError(
            f"ambiguous {role!r} column: {hits}. Pass it explicitly "
            f"(columns seen: {list(df.columns)})")
    return None


def infer_fs(time_values) -> tuple[float, str]:
    """Sampling rate from the time column. Returns (fs, unit_guess)."""
    t = np.unique(np.asarray(time_values, dtype=float))
    if t.size < 2:
        raise IngestError("cannot infer fs from fewer than 2 distinct times")
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        raise IngestError(f"non-increasing time column (median dt={dt})")
    unit = "ms" if np.nanmax(np.abs(t)) > 10.0 else "s"
    fs = (1000.0 / dt) if unit == "ms" else (1.0 / dt)
    if not (1.0 < fs < 100000.0):
        raise IngestError(f"inferred fs={fs:.3f} Hz is implausible "
                          f"(dt={dt}, unit={unit})")
    return fs, unit


def canonical_condition(token, overrides: dict[str, str] | None = None) -> str:
    """Map a raw condition token to one of CONDITIONS."""
    if overrides:
        for canon, raw in overrides.items():
            if _norm(token) == _norm(raw):
                if canon not in CONDITIONS:
                    raise IngestError(f"override target {canon!r} not in "
                                      f"{CONDITIONS}")
                return canon
    n = _norm(token)
    for canon, tokens in CONDITION_TOKENS.items():
        if n in tokens:
            return canon
    raise IngestError(
        f"unrecognised condition token {token!r}; pass "
        f"--condition-map '{{\"selfsz\":\"{token}\"}}' (known: "
        f"{sorted(CONDITION_TOKENS)})")


@dataclass
class EpochSet:
    """One (condition) block: arrays ready for the on-disk layout."""
    by_subject: dict[str, np.ndarray] = field(default_factory=dict)
    n_samples: int = 0
    fs: float = 0.0
    time: np.ndarray | None = None
    onset_s: float = 0.0


def stack_long(df: pd.DataFrame, cols: dict, overrides=None) -> dict[str, EpochSet]:
    """Long layout (subject, condition, electrode, time, value) -> per-subject
    (channels, time) arrays, channels in CHANNELS order."""
    out: dict[str, EpochSet] = {}
    d = df.copy()
    d["_cond"] = [canonical_condition(v, overrides) for v in d[cols["condition"]]]
    d["_subj"] = d[cols["subject"]].astype(str).str.strip()

    seen_electrodes = {_norm(v) for v in d[cols["electrode"]].unique()}
    missing = [c for c in CHANNELS if _norm(c) not in seen_electrodes]
    if missing:
        raise IngestError(
            f"electrodes missing from the file: {missing}. Present: "
            f"{sorted(d[cols['electrode']].unique())[:30]}. Extend CHANNELS in "
            "src/data.py only if you can cite why.")

    for (cond, subj), g in d.groupby(["_cond", "_subj"], sort=True):
        t = np.unique(g[cols["time"]].to_numpy(dtype=float))
        fs, unit = infer_fs(t)
        onset = -float(t[0]) / (1000.0 if unit == "ms" else 1.0)
        mat = np.full((len(CHANNELS), t.size), np.nan, dtype=np.float32)
        pos = {v: i for i, v in enumerate(t)}
        for elec, ge in g.groupby(cols["electrode"]):
            row = next(i for i, c in enumerate(CHANNELS)
                       if _norm(c) == _norm(elec))
            tv = ge[cols["time"]].to_numpy(dtype=float)
            vv = ge[cols["value"]].to_numpy(dtype=float)
            if tv.size != np.unique(tv).size:
                raise IngestError(
                    f"{cond}/{subj}/{elec}: duplicate time points - refusing to "
                    "average them silently")
            mat[row, [pos[x] for x in tv]] = vv
        if not np.isfinite(mat).all():
            raise IngestError(f"{cond}/{subj}: incomplete time grid")
        ep = out.setdefault(cond, EpochSet(n_samples=t.size, fs=fs, time=t,
                                          onset_s=onset))
        if abs(ep.onset_s - onset) > 1e-6:
            raise IngestError(
                f"{cond}/{subj}: epoch starts at a different offset "
                f"({onset:.4f}s vs {ep.onset_s:.4f}s) - the event would not be "
                "aligned across subjects in this condition")
        if ep.n_samples != t.size or abs(ep.fs - fs) > 1e-6:
            raise IngestError(
                f"{cond}/{subj}: epoch length {t.size}@ {fs:g}Hz differs from "
                f"{ep.n_samples}@ {ep.fs:g}Hz of other subjects in the same "
                "condition")
        ep.by_subject[str(subj)] = mat
    return out


def stack_wide(df: pd.DataFrame, cols: dict, overrides=None) -> dict[str, EpochSet]:
    """Wide layout (time + one column per electrode) -> same output."""
    out: dict[str, EpochSet] = {}
    have = {_norm(c): c for c in df.columns}
    missing = [c for c in CHANNELS if _norm(c) not in have]
    if missing:
        raise IngestError(f"wide layout is missing electrode columns: {missing}")
    d = df.copy()
    d["_cond"] = [canonical_condition(v, overrides) for v in d[cols["condition"]]]
    d["_subj"] = d[cols["subject"]].astype(str).str.strip()
    for (cond, subj), g in d.groupby(["_cond", "_subj"], sort=True):
        g = g.sort_values(cols["time"])
        t = g[cols["time"]].to_numpy(dtype=float)
        if np.unique(t).size != t.size:
            raise IngestError(f"{cond}/{subj}: duplicate time points")
        fs, unit = infer_fs(t)
        onset = -float(t[0]) / (1000.0 if unit == "ms" else 1.0)
        mat = g[[have[_norm(c)] for c in CHANNELS]].to_numpy(dtype=np.float32).T
        ep = out.setdefault(cond, EpochSet(n_samples=t.size, fs=fs, time=t,
                                          onset_s=onset))
        if ep.n_samples != t.size or abs(ep.fs - fs) > 1e-6:
            raise IngestError(f"{cond}/{subj}: inconsistent epoch grid")
        ep.by_subject[str(subj)] = mat
    return out


def read_demographics(path, subject_col=None, label_col=None,
                      scz_values=None) -> dict[str, int]:
    """demographic.csv -> {subject: 1|0}. Unknown values raise, never default."""
    path = Path(path)
    if not path.exists():
        raise IngestError(f"demographics file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    scol = subject_col or detect_column(df, "subject")
    lcol = label_col or detect_column(df, "label")
    if scol is None or lcol is None:
        raise IngestError(
            f"cannot find subject/label columns in {path.name}; columns: "
            f"{list(df.columns)}. Pass --subject-col and --label-col.")
    scz = {_norm(v) for v in (scz_values or SCZ_TOKENS)}
    out: dict[str, int] = {}
    unknown = set()
    for sid, raw in zip(df[scol].astype(str).str.strip(), df[lcol]):
        if pd.isna(raw):
            continue
        token = _norm(raw)
        if token in scz:
            lab = 1
        elif token in {"0", "hc", "control", "controls", "healthy", "comparison",
                       "con", "normal", "nc"}:
            lab = 0
        else:
            unknown.add(str(raw))
            continue
        if sid in out and out[sid] != lab:
            raise IngestError(f"{sid}: conflicting labels in demographics")
        out[sid] = lab
    if unknown:
        raise IngestError(
            f"unmapped diagnosis values {sorted(unknown)} in {path.name}; pass "
            f"--scz-values to name the patient values explicitly")
    if not out:
        raise IngestError(f"{path.name}: no labelled rows")
    return out


def write_dataset(epochs: dict[str, EpochSet], labels: dict[str, int], out_dir,
                  meta_extra: dict | None = None) -> dict:
    """Write data/<cond>/<cond>_<subject>.parquet + labels.csv + meta."""
    out = Path(out_dir)
    written, per_cond = 0, {}
    for cond, ep in sorted(epochs.items()):
        subj_keys = sorted(ep.by_subject)
        absent = [s for s in subj_keys if s not in labels]
        if absent:
            raise IngestError(
                f"{cond}: {len(absent)} subjects have no label, e.g. "
                f"{absent[:5]}. Refusing to write a partially labelled set.")
        d = out / cond
        d.mkdir(parents=True, exist_ok=True)
        for sid in subj_keys:
            frame = pd.DataFrame(ep.by_subject[sid].T, columns=CHANNELS)
            frame.to_parquet(d / f"{cond}_{sid}.parquet", index=False)
            written += 1
        per_cond[cond] = {"subjects": len(subj_keys),
                          "n_samples": ep.n_samples, "fs": ep.fs,
                          "event_onset_s": ep.onset_s}
    all_subj = sorted({s for ep in epochs.values() for s in ep.by_subject})
    rows = ["subject_id,label"] + [f"{s},{labels[s]}" for s in all_subj]
    (out / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    fs_set = {round(v["fs"], 6) for v in per_cond.values()}
    if len(fs_set) != 1:
        raise IngestError(f"conditions disagree on fs: {fs_set}")
    n_set = {v["n_samples"] for v in per_cond.values()}
    if len(n_set) != 1:
        raise IngestError(f"conditions disagree on epoch length: {n_set}")
    onset_set = {round(v["event_onset_s"], 6) for v in per_cond.values()}
    if len(onset_set) != 1:
        raise IngestError(f"conditions disagree on event onset: {onset_set}")
    meta = {"fs": float(next(iter(fs_set))),
            "n_samples": int(next(iter(n_set))),
            "channels": CHANNELS,
            "epoch_seconds": float(next(iter(n_set)) / next(iter(fs_set))),
            "event_onset_s": float(next(iter(onset_set))),
            "per_condition": per_cond,
            "n_subjects": len(all_subj),
            "n_scz": int(sum(labels[s] for s in all_subj)),
            "n_control": int(len(all_subj) - sum(labels[s] for s in all_subj))}
    meta.update(meta_extra or {})
    (out / "dataset_meta.json").write_text(
        __import__("json").dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return {"written_files": written, "meta": meta}
