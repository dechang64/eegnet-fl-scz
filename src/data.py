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

# The nine sites the dataset's own description and the replications analyse
# (Fz FCz Cz FC3 FC4 C3 C4 CP3 CP4). Order is the one the dataset text lists;
# it is NOT guaranteed to be the column order of any given export, which is why
# build_dataset.py selects columns by name and refuses on a missing site.
CHANNELS = ["Fz", "FCz", "Cz", "FC3", "C3", "CP3", "FC4", "C4", "CP4"]

# Approximate 10-20 positions on a unit scalp disc: x lateral (- = left/odd),
# y anterior. Approximate is fine for a weight map; it is not a projection and
# must not be used to claim anatomical localisation.
CHANNEL_POS = {
    "Fz": (0.00, 0.62), "FCz": (0.00, 0.38), "Cz": (0.00, 0.00),
    "FC3": (-0.42, 0.38), "C3": (-0.62, 0.00), "CP3": (-0.42, -0.38),
    "FC4": (0.42, 0.38), "C4": (0.62, 0.00), "CP4": (0.42, -0.38),
}
# Kept for callers written against the old (wrong: 5 midline channels) name.
MIDLINE_POS = CHANNEL_POS

SUPPORTED_SUFFIXES = (".parquet", ".npy")
MIN_TIMESTEPS = 32
META_NAME = "dataset_meta.json"


def resolve_event_onset(data_dir) -> float:
    """Seconds from the start of the epoch to the event (stimulus / button).

    button-tone-sz epochs are centred on the event (-1.5 s .. +1.5 s), so the
    event sits at t=1.5 s, not t=0. The loader keeps only the samples and drops
    the time column, so without this the N100 window (80-150 ms) would be
    shaded near the start of a centred epoch: correct-looking, wrong by 1.5 s.
    The ingester derives it from the real time column.
    """
    import json
    meta_path = Path(data_dir) / META_NAME
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} missing: event onset is unknown, so no latency may "
            "be read off a time axis. Build the dataset with "
            "scripts/build_dataset.py first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "event_onset_s" not in meta:
        raise KeyError(
            f"{meta_path}: 'event_onset_s' absent (built by an older "
            "ingester). Rebuild it - do not assume 0.")
    return float(meta["event_onset_s"])


def resolve_fs(data_dir) -> float:
    """Sampling rate, read from `dataset_meta.json` written by the ingester.

    Deliberately not a constant. Every time axis in xai_dissect.py is
    `np.arange(T) / FS`; a wrong constant (the old guess was 500 Hz) would put
    the N100 band at the wrong place on every figure while still plotting
    cleanly - the quietest possible corruption. The ingester derives it from
    the real time column, so the only honest source is that file.
    """
    import json
    meta_path = Path(data_dir) / META_NAME
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} missing: sampling rate is unknown, so no time axis "
            "may be drawn. Build the dataset with scripts/build_dataset.py "
            "(which derives fs from the real time column) first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fs = float(meta["fs"])
    if not (0.0 < fs < 100000.0):
        raise ValueError(f"{meta_path}: implausible fs={fs}")
    return fs


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
