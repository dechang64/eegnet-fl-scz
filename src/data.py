"""button-tone-sz loader.

AUDIT 2026-11-10
---------------
B3 (high, latent): `_subject_of` returned `stem.split("_")[0]`, i.e. for
    `selfsz_s01.npy` it returned "selfsz" - every subject collapsed into the
    condition name, turning 81 subjects into 3 pseudo-subjects and making the
    group split a no-op (total leakage). It was masked only because the label
    guard raised first, so the first D1 fix to labelling would have silently
    destroyed the protocol. Subject ids are now parsed and validated, and a
    degenerate parse raises.
B3b (high): labels no longer come from filename heuristics at all. They come
    from an explicit `--labels` CSV (subject_id,label). Without that file the
    loader refuses to guess.
B4 (med): array shape/channel-count/finiteness are validated; the previous
    `reshape(arr.shape[-2], -1)` silently reinterpreted 1-D or 3-D files.

Design: FAIL LOUDLY. Every unknown is an exception with an instruction, never
a default that a reader could mistake for a finding.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = ("selfsz", "passivesz", "buttonsz")
CHANNELS = ["Fz", "FCz", "Cz", "CPz", "Pz"]        # TODO(D1): verify
FS = 500.0                                          # TODO(D1): verify
MIDLINE_POS = {"Fz": (0.0, 0.62), "FCz": (0.0, 0.38), "Cz": (0.0, 0.0),
               "CPz": (0.0, -0.38), "Pz": (0.0, -0.62)}
SUPPORTED_SUFFIXES = (".parquet", ".npy")
MIN_TIMESTEPS = 32


def _load_array(fp: Path) -> np.ndarray:
    """Return (channels, time) or (epochs, channels, time) float32.

    Never guesses a layout silently. 2-D means one epoch, 3-D means a stack of
    epochs for that subject; anything else raises.
    """
    if fp.suffix == ".parquet":
        df = pd.read_parquet(fp)
        if df.ndim != 2:
            raise ValueError(f"{fp.name}: parquet must be 2-D, got {df.ndim}")
        # long format (columns = channels) is transposed; wide/samples layout is
        # detected by which axis matches the known channel names
        named = [c for c in df.columns if c in CHANNELS]
        arr = df.to_numpy(dtype=np.float32)
        if len(named) >= 2:
            return arr.T
        return arr
    if fp.suffix == ".npy":
        arr = np.asarray(np.load(fp), dtype=np.float32)
        if arr.ndim not in (2, 3):
            raise ValueError(
                f"{fp.name}: expected 2-D (channels, time) or 3-D "
                f"(epochs, channels, time), got shape {arr.shape}. "
                "Refusing to reshape - fix the export instead.")
        return arr
    raise NotImplementedError(
        f"Unsupported file type: {fp.name}. Extend _load_array() at D1 "
        f"(supported: {SUPPORTED_SUFFIXES}).")


def parse_stem(fp: Path) -> tuple[str, str]:
    """`<condition>_<subject>[_anything]` -> (condition, subject).

    Raises rather than returning a plausible-looking wrong id.
    """
    parts = fp.stem.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"{fp.name}: cannot parse subject id from stem {fp.stem!r}; "
            "expected '<condition>_<subject>...' - extend parse_stem() at D1")
    cond, subject = parts[0].lower(), parts[1]
    if cond not in CONDITIONS:
        raise ValueError(
            f"{fp.name}: leading token {parts[0]!r} is not one of "
            f"{CONDITIONS}; extend CONDITIONS at D1")
    if not subject or subject.lower() in CONDITIONS:
        raise ValueError(
            f"{fp.name}: degenerate subject id {subject!r} (B3 guard) - the "
            "whole cohort would collapse into one group")
    return cond, subject


def read_labels(labels_csv) -> dict[str, int]:
    """CSV with header `subject_id,label`; label 1 = SCZ, 0 = control."""
    if labels_csv is None:
        raise NotImplementedError(
            "load_condition() needs ground-truth labels: pass labels_csv "
            "(subject_id,label with 1=SCZ, 0=control). Filenames in this "
            "dataset embed condition names, so any filename-based label "
            "heuristic is wrong by construction (H1).")
    path = Path(labels_csv)
    if not path.exists():
        raise FileNotFoundError(f"labels file not found: {path}")
    out: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("subject_id") or "").strip()
            raw = (row.get("label") or "").strip()
            if not sid or raw == "":
                raise ValueError(f"{path.name}: malformed row {row}")
            lab = int(raw)
            if lab not in (0, 1):
                raise ValueError(
                    f"{path.name}: label must be 0/1, got {lab} for {sid!r}")
            if sid in out and out[sid] != lab:
                raise ValueError(
                    f"{path.name}: subject {sid!r} listed with conflicting "
                    f"labels {out[sid]} and {lab}")
            out[sid] = lab
    if not out:
        raise ValueError(f"{path.name}: no label rows parsed")
    return out


def load_condition(data_dir, cond: str, labels_csv=None,
                   n_channels: int | None = None):
    """Load one condition.

    Returns (X, subjects, y) with one row per EPOCH:
      X: (n_epochs, channels, time) float32
      subjects: (n_epochs,) subject id per epoch  <- the CV groups
      y: (n_epochs,) 1 = SCZ, 0 = control
    A 2-D file contributes one epoch, a 3-D file contributes arr.shape[0].
    """
    if cond not in CONDITIONS:
        raise ValueError(f"unknown condition {cond!r}; use one of {CONDITIONS}")
    labels = read_labels(labels_csv)          # R2: label guard BEFORE parsing
    cond_dir = Path(data_dir) / cond
    if not cond_dir.exists():
        raise FileNotFoundError(f"{cond_dir} not found")
    want_c = len(CHANNELS) if n_channels is None else int(n_channels)

    X, subj, y = [], [], []
    t_ref = None
    for fp in sorted(cond_dir.iterdir()):
        if fp.suffix not in SUPPORTED_SUFFIXES:
            continue
        parsed_cond, subject = parse_stem(fp)
        if parsed_cond != cond:
            raise ValueError(
                f"{fp.name}: condition token {parsed_cond!r} does not match "
                f"its directory {cond!r}")
        if subject not in labels:
            raise KeyError(
                f"{fp.name}: subject {subject!r} missing from labels CSV")
        arr = _load_array(fp)
        epochs = arr if arr.ndim == 3 else arr[None]
        if epochs.shape[1] != want_c:
            raise ValueError(
                f"{fp.name}: expected {want_c} channels "
                f"({CHANNELS}), got {epochs.shape[1]}. Channel layout must be "
                "confirmed at D1 before any number is trusted.")
        if epochs.shape[2] < MIN_TIMESTEPS:
            raise ValueError(
                f"{fp.name}: only {epochs.shape[2]} samples per epoch "
                f"(< {MIN_TIMESTEPS})")
        if not np.isfinite(epochs).all():
            raise ValueError(f"{fp.name}: contains NaN/inf")
        if t_ref is None:
            t_ref = epochs.shape[2]
        elif epochs.shape[2] != t_ref:
            raise ValueError(
                f"{fp.name}: epoch length {epochs.shape[2]} != {t_ref} of the "
                "first file in this condition")
        X.append(epochs)
        subj.extend([subject] * len(epochs))
        y.extend([labels[subject]] * len(epochs))
    if not X:
        raise RuntimeError(f"no supported files in {cond_dir}")
    if len(set(y)) < 2:
        raise RuntimeError(
            f"degenerate labels {sorted(set(y))} in {cond_dir} - check the "
            "labels CSV (both classes must be present)")
    return (np.concatenate(X, axis=0).astype(np.float32),
            np.asarray(subj), np.asarray(y, dtype=np.int64))
