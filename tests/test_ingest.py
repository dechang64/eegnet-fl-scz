"""Ingest guards: every test maps to a way the converter could quietly lie.

The converter is the single point where the raw Kaggle export becomes the
arrays an AUC is computed from. If it silently averages duplicates, drops an
electrode, guesses a diagnosis value or invents a sampling rate, every
downstream number is meaningless while still looking clean. So these tests do
not check "does it run" - they check "does it refuse".
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data import CHANNELS, MIN_TIMESTEPS, META_NAME, resolve_fs
from src.ingest import (IngestError, canonical_condition, detect_column,
                        infer_fs, read_demographics, stack_long, stack_wide,
                        write_dataset)

N_T = max(MIN_TIMESTEPS, 60)


def long_frame(subjects=("s1", "s2"), conds=("selfsz", "passivesz"),
               electrodes=tuple(CHANNELS), n_t=N_T, fs=500.0, seed=0):
    """Well-formed long-format frame, the shape ERPdata.csv is expected to have."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_t) / fs
    rows = []
    for sid in subjects:
        for cond in conds:
            for elec in electrodes:
                wave = rng.normal(0, 1, n_t)
                rows.extend({"subject": sid, "condition": cond,
                             "electrode": elec, "time": float(tt),
                             "value": float(v)}
                            for tt, v in zip(t, wave))
    return pd.DataFrame(rows)


def label_file(tmp_path, mapping, name="demographic.csv"):
    p = tmp_path / name
    p.write_text("subject,diagnosis\n"
                 + "\n".join(f"{k},{v}" for k, v in mapping.items()) + "\n",
                 encoding="utf-8")
    return p


# --------------------------------------------------------------- sampling rate
def test_infer_fs_seconds():
    fs, unit = infer_fs(np.arange(N_T) / 500.0)
    assert fs == pytest.approx(500.0) and unit == "s"


def test_infer_fs_milliseconds():
    fs, unit = infer_fs(np.arange(-750, 750, 2.0))   # -1.5 s .. +1.5 s @ 500 Hz
    assert unit == "ms" and fs == pytest.approx(500.0)


def test_infer_fs_rejects_flat_time():
    with pytest.raises(IngestError, match="fewer than 2"):
        infer_fs([0.0, 0.0, 0.0])


def test_resolve_fs_raises_without_meta(tmp_path):
    with pytest.raises(FileNotFoundError, match=META_NAME):
        resolve_fs(tmp_path)


def test_resolve_event_onset_from_meta(tmp_path):
    from src.data import resolve_event_onset
    (tmp_path / META_NAME).write_text(
        json.dumps({"fs": 250.0, "event_onset_s": 1.5}), encoding="utf-8")
    assert resolve_event_onset(tmp_path) == pytest.approx(1.5)


def test_resolve_event_onset_refuses_missing_key(tmp_path):
    from src.data import resolve_event_onset
    (tmp_path / META_NAME).write_text(json.dumps({"fs": 250.0}),
                                      encoding="utf-8")
    with pytest.raises(KeyError, match="event_onset_s"):
        resolve_event_onset(tmp_path)


def test_onset_is_derived_from_centred_epoch(tmp_path):
    """A -1.5 s .. +1.5 s epoch must report an onset of 1.5 s, not 0.

    This is the difference between the N100 band landing at 100 ms and at
    -1.4 s on every virtual-ERP figure.
    """
    fs, n = 250.0, 750                      # 3 s epoch, as in the dataset
    t = (np.arange(n) - n // 2) / fs        # starts at -1.496 s
    df = long_frame(subjects=("s1",), conds=("selfsz",), n_t=n, fs=fs)
    df["time"] = np.tile(t, len(df) // n)[: len(df)]
    ep = stack_long(df, {"subject": "subject", "condition": "condition",
                         "electrode": "electrode", "time": "time",
                         "value": "value"})
    assert ep["selfsz"].onset_s == pytest.approx((n // 2) / fs, abs=1e-6)
    assert ep["selfsz"].onset_s > 1.4       # i.e. not silently 0


def test_write_records_onset_in_meta(tmp_path):
    ep = stack_long(long_frame(), {"subject": "subject",
                                   "condition": "condition",
                                   "electrode": "electrode",
                                   "time": "time", "value": "value"})
    res = write_dataset(ep, {"s1": 1, "s2": 0}, tmp_path)
    assert res["meta"]["event_onset_s"] == pytest.approx(0.0, abs=1e-6)


def test_resolve_fs_reads_meta(tmp_path):
    (tmp_path / META_NAME).write_text(json.dumps({"fs": 500.0}), encoding="utf-8")
    assert resolve_fs(tmp_path) == pytest.approx(500.0)


# ------------------------------------------------------------- column detection
def test_detect_column_exact_and_alias():
    df = pd.DataFrame({"SubjectID": [1], "Condition": ["self"],
                       "Electrode": ["Fz"], "Time": [0.0], "Amplitude": [1.0]})
    assert detect_column(df, "subject") == "SubjectID"
    assert detect_column(df, "value") == "Amplitude"
    assert detect_column(df, "electrode") == "Electrode"


def test_detect_column_ambiguous_raises():
    # No alias matches exactly, two match by substring -> must not pick one.
    df = pd.DataFrame({"subjectcode": [1], "subjectref": [1]})
    with pytest.raises(IngestError, match="ambiguous"):
        detect_column(df, "subject")


def test_detect_column_exact_alias_beats_substring():
    # A defined priority is not an ambiguity: 'subject' wins over 'subject_id'.
    df = pd.DataFrame({"subject": [1], "subject_id": [2]})
    assert detect_column(df, "subject") == "subject"


def test_detect_column_absent_returns_none():
    assert detect_column(pd.DataFrame({"foo": [1]}), "electrode") is None


# ------------------------------------------------------------------ conditions
def test_condition_tokens_and_override():
    assert canonical_condition("1") == "selfsz"
    assert canonical_condition("PASSIVE") == "passivesz"
    assert canonical_condition("button") == "buttonsz"
    assert canonical_condition("selftone", {"selfsz": "selftone"}) == "selfsz"
    with pytest.raises(IngestError, match="unrecognised condition"):
        canonical_condition("mystery")


# ---------------------------------------------------------------------- labels
def test_demographics_maps_patient_and_control(tmp_path):
    f = label_file(tmp_path, {"s1": "SZ", "s2": "Control", "s3": "HC"})
    assert read_demographics(f) == {"s1": 1, "s2": 0, "s3": 0}


def test_demographics_unmapped_value_raises(tmp_path):
    f = label_file(tmp_path, {"s1": "SZ", "s2": "bipolar"})
    with pytest.raises(IngestError, match="unmapped diagnosis"):
        read_demographics(f)


def test_demographics_explicit_scz_values(tmp_path):
    f = label_file(tmp_path, {"s1": "bipolar", "s2": "control"})
    assert read_demographics(f, scz_values=["bipolar"])["s1"] == 1


# ------------------------------------------------------------------- stacking
def test_stack_long_orders_channels_as_CHANNELS():
    ep = stack_long(long_frame(), {"subject": "subject",
                                   "condition": "condition",
                                   "electrode": "electrode",
                                   "time": "time", "value": "value"})
    assert set(ep) == {"selfsz", "passivesz"}
    mat = ep["selfsz"].by_subject["s1"]
    assert mat.shape == (len(CHANNELS), N_T)
    assert ep["selfsz"].fs == pytest.approx(500.0)
    assert ep["selfsz"].n_samples == N_T


def test_stack_long_missing_electrode_raises():
    df = long_frame(electrodes=tuple(CHANNELS[:-1]))
    with pytest.raises(IngestError, match="electrodes missing"):
        stack_long(df, {"subject": "subject", "condition": "condition",
                        "electrode": "electrode", "time": "time",
                        "value": "value"})


def test_stack_long_duplicate_time_raises():
    df = long_frame(subjects=("s1",), conds=("selfsz",))
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(IngestError, match="duplicate time"):
        stack_long(dup, {"subject": "subject", "condition": "condition",
                         "electrode": "electrode", "time": "time",
                         "value": "value"})


def test_stack_long_inconsistent_epoch_length_raises():
    a = long_frame(subjects=("s1",), conds=("selfsz",))
    b = long_frame(subjects=("s2",), conds=("selfsz",), n_t=N_T + 5)
    both = pd.concat([a, b], ignore_index=True)
    with pytest.raises(IngestError, match="differs"):
        stack_long(both, {"subject": "subject", "condition": "condition",
                          "electrode": "electrode", "time": "time",
                          "value": "value"})


def test_stack_wide_layout():
    t = np.arange(N_T) / 500.0
    rows = []
    for sid in ("s1", "s2"):
        for cond in ("selfsz",):
            rec = {"subject": sid, "condition": cond}
            for c in CHANNELS:
                rec[c] = 0.0
            for i, tt in enumerate(t):
                row = dict(rec, time=float(tt))
                for c in CHANNELS:
                    row[c] = float(np.sin(i / 7.0))
                rows.append(row)
    df = pd.DataFrame(rows)
    ep = stack_wide(df, {"subject": "subject", "condition": "condition",
                         "time": "time"})
    assert ep["selfsz"].by_subject["s1"].shape == (len(CHANNELS), N_T)


# ------------------------------------------------------------------ round trip
def test_write_then_load_round_trip(tmp_path):
    """The whole point: what the converter writes, src/data.py can read back."""
    from src.data import load_condition
    df = long_frame()
    ep = stack_long(df, {"subject": "subject", "condition": "condition",
                         "electrode": "electrode", "time": "time",
                         "value": "value"})
    labels = {"s1": 1, "s2": 0}
    res = write_dataset(ep, labels, tmp_path)
    assert res["written_files"] == 4
    assert res["meta"]["fs"] == pytest.approx(500.0)
    assert res["meta"]["n_scz"] == 1

    X, subj, y = load_condition(tmp_path, "selfsz",
                                labels_csv=tmp_path / "labels.csv")
    assert X.shape == (2, len(CHANNELS), N_T)
    assert sorted(set(subj.tolist())) == ["s1", "s2"]
    assert set(y.tolist()) == {0, 1}


def test_write_refuses_unlabelled_subject(tmp_path):
    df = long_frame()
    ep = stack_long(df, {"subject": "subject", "condition": "condition",
                         "electrode": "electrode", "time": "time",
                         "value": "value"})
    with pytest.raises(IngestError, match="no label"):
        write_dataset(ep, {"s1": 1}, tmp_path)


def test_write_refuses_mixed_fs(tmp_path):
    a = stack_long(long_frame(subjects=("s1",), conds=("selfsz",)),
                   {"subject": "subject", "condition": "condition",
                    "electrode": "electrode", "time": "time", "value": "value"})
    b = stack_long(long_frame(subjects=("s1",), conds=("passivesz",), fs=250.0),
                   {"subject": "subject", "condition": "condition",
                    "electrode": "electrode", "time": "time", "value": "value"})
    a.update(b)
    with pytest.raises(IngestError, match="disagree on fs"):
        write_dataset(a, {"s1": 1}, tmp_path)
