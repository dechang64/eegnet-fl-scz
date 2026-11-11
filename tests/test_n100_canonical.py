"""Regression guard for the N100 metric (AUDIT.md Addendum 3).

The audit claims that the canonical metric - 80-100 ms mean at Fz/FCz/Cz, with
the button-only condition subtracted - reproduces the dataset author's numbers
(HC 2.2 uV, SZ 0.88 uV), and that dropping either the window or the subtraction
destroys the group contrast. Both halves of that claim are tested here, because
the second half is what makes the first one informative.

Skipped when the real data is absent: `data/` never enters git, so CI has no
epochs to read. Locally this runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DATA = ROOT / "data"
pytestmark = pytest.mark.skipif(
    not (DATA / "dataset_meta.json").exists(),
    reason="real epochs not present (data/ is gitignored)",
)


def _run():
    import n100_canonical as m  # noqa: E402

    meta = json.loads((DATA / "dataset_meta.json").read_text(encoding="utf-8"))
    fs, onset = meta["fs"], meta["event_onset_s"]
    import pandas as pd  # noqa: E402

    labels = pd.read_csv(DATA / "labels.csv")
    lab = dict(zip(labels.subject_id.astype(str).str.replace(r"\.0$", "", regex=True),
                   labels.label))
    rows = []
    for sid in lab:
        s = m.window_amp(DATA, "selfsz", sid, fs, onset)
        pvp = m.window_amp(DATA, "passivesz", sid, fs, onset)
        b = m.window_amp(DATA, "buttonsz", sid, fs, onset)
        rows.append((lab[sid], pvp - (s - b), pvp - s))
    d = pd.DataFrame(rows, columns=["label", "corrected", "naive"])
    return d


def test_canonical_metric_reproduces_published_numbers():
    d = _run()
    hc = d[d.label == 0].corrected
    sz = d[d.label == 1].corrected
    assert len(hc) == 32 and len(sz) == 49, "cohort split must be 32 HC / 49 SZ"
    # Roach's notebook: 2.2 uV in HC, 0.88 uV in SZ. Tolerance is 0.15 uV -
    # wide enough for a different MNE/pandas rounding, tight enough that a
    # wrong window or a missing subtraction cannot pass.
    assert abs(hc.mean() - (-2.20)) < 0.15, f"HC suppression {hc.mean():.3f}"
    assert abs(sz.mean() - (-0.88)) < 0.15, f"SZ suppression {sz.mean():.3f}"


def test_group_contrast_is_significant_only_with_subtraction():
    d = _run()
    hc, sz = d[d.label == 0].corrected, d[d.label == 1].corrected
    assert stats.ttest_ind(hc, sz, equal_var=False).pvalue < 0.01
    hc2, sz2 = d[d.label == 0].naive, d[d.label == 1].naive
    assert stats.ttest_ind(hc2, sz2, equal_var=False).pvalue > 0.05, (
        "without the button-only subtraction the contrast should collapse - "
        "if it no longer does, Addendum 3's explanation is wrong"
    )


def test_suppression_is_negative_in_both_groups():
    """The N100 is a negative deflection, so playback must be more negative
    than the corrected button+tone epoch in both groups."""
    d = _run()
    for label in (0, 1):
        assert d[d.label == label].corrected.mean() < 0
