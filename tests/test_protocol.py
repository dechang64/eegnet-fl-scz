"""Protocol / model / loader guards.

Every test here maps to an audit finding (AUDIT.md). A test that fails means a
claim in the README is unsupported - which is the point.
"""
import warnings

import numpy as np
import pytest

from src.protocol import SubjectBasedCV, aggregate_folds, report_metrics


# ---------------------------------------------------------------- CV protocol
def test_no_subject_overlap():
    subj = np.repeat([f"s{i:02d}" for i in range(81)], 10)
    y = np.repeat(np.repeat([0, 1], 41)[:81], 10)   # one label per subject
    for tr, te in SubjectBasedCV().split(subj, y):
        assert set(subj[tr]).isdisjoint(subj[te])


def test_all_subjects_used_exactly_once():
    subj = np.repeat([f"s{i:02d}" for i in range(20)], 5)
    y = np.repeat(np.repeat([0, 1], 10), 5)
    seen = []
    for _, te in SubjectBasedCV().split(subj, y):
        seen.extend(subj[te].tolist())
    assert sorted(seen) == sorted(subj.tolist())


def test_leakage_guard_raises(monkeypatch):
    subj = np.repeat(["s0", "s1"], 5)
    y = np.repeat([0, 1], 5)
    cv = SubjectBasedCV(n_splits=2)

    def fake_split(X, y_, groups=None):
        yield np.arange(0, 10), np.arange(0, 10)

    monkeypatch.setattr(cv.cv, "split", fake_split)
    with pytest.raises(RuntimeError, match="LEAKAGE"):
        list(cv.split(subj, y))


def test_conflicting_subject_labels_raise():
    subj = np.array(["s0", "s0", "s1", "s1"] * 3)
    y = np.array([0, 1, 0, 1] * 3)          # s0 carries both labels -> fault
    with pytest.raises(RuntimeError, match="conflicting labels"):
        list(SubjectBasedCV(n_splits=2).split(subj, y))


def test_too_few_subjects_raise():
    with pytest.raises(ValueError, match="subjects for"):
        list(SubjectBasedCV(n_splits=5).split(np.zeros(4), np.zeros(4)))


# --------------------------------------------------------------------- metrics
def test_metrics_report():
    m = report_metrics([0, 0, 1, 1, 1, 0, 1, 0],
                       [0, 1, 1, 1, 0, 0, 1, 0],
                       [.1, .6, .8, .9, .4, .2, .7, .3])
    assert 0.0 <= m["acc"] <= 1.0
    assert m["boot_units"] == 8


def test_single_class_metrics_are_nan_not_crash():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = report_metrics([1, 1, 1], [1, 0, 1], [.9, .4, .8])
    assert np.isnan(m["auc"])
    assert np.isnan(m["auc_ci"][0])
    assert m["single_class"] is True


def test_metrics_without_groups_warns():
    with pytest.warns(RuntimeWarning, match="anti-conservative"):
        report_metrics([0, 1, 0, 1], [0, 1, 0, 1], [.1, .9, .2, .8])


def test_cluster_bootstrap_is_not_narrower_than_epoch_bootstrap():
    """B2 regression: epoch bootstrap ignores subject clustering."""
    rng = np.random.default_rng(0)
    n_subj, per_subj = 20, 10
    subj = np.repeat(np.array([f"s{i}" for i in range(n_subj)]), per_subj)
    y = np.repeat(np.array([0, 1] * (n_subj // 2)), per_subj)
    subj_effect = rng.normal(0, 2.0, n_subj)
    score = (subj_effect[np.repeat(np.arange(n_subj), per_subj)]
             + y * 0.5 + rng.normal(0, 0.1, n_subj * per_subj))
    pred = (score > np.median(score)).astype(int)
    cluster = report_metrics(y, pred, score, groups=subj, n_boot=400, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        epoch = report_metrics(y, pred, score, n_boot=400, seed=1)
    w_cluster = cluster["auc_ci"][1] - cluster["auc_ci"][0]
    w_epoch = epoch["auc_ci"][1] - epoch["auc_ci"][0]
    assert w_cluster > w_epoch, (w_cluster, w_epoch)


def test_aggregate_folds_weighted():
    folds = [dict(acc=.9, auc=.9, sens=.9, spec=.9, f1=.9),
             dict(acc=.7, auc=.7, sens=.7, spec=.7, f1=.7)]
    agg = aggregate_folds(folds, [40, 4])
    assert agg["auc"][0] == pytest.approx(0.8)
    assert agg["auc_subject_weighted"][0] == pytest.approx(
        (.9 * 40 + .7 * 4) / 44)


# ---------------------------------------------------------------------- models
def test_eegnet_forward_shape():
    torch = pytest.importorskip("torch")
    from src.eegnet import EEGNet
    for T in (31, 32, 63, 97, 100, 320, 500):
        m = EEGNet(n_channels=5, n_timesteps=T)
        assert m(torch.randn(2, 1, 5, T)).shape == (2, 2)


def test_feat_t_regression_b1():
    """B1: lengths where the old closed form disagreed with reality."""
    torch = pytest.importorskip("torch")
    from src.eegnet import EEGNet, pooled_feat_t
    for T in (27, 28, 29, 30, 59, 60, 61, 62, 91, 92, 93, 94):
        m = EEGNet(n_channels=5, n_timesteps=T)
        assert m(torch.randn(2, 1, 5, T)).shape == (2, 2)
        assert m.feat_t == pooled_feat_t(T)


def test_eegnet_rejects_wrong_input_shape():
    torch = pytest.importorskip("torch")
    from src.eegnet import EEGNet
    m = EEGNet(n_channels=5, n_timesteps=64)
    with pytest.raises(ValueError, match="expected input"):
        m(torch.randn(2, 5, 64))
    with pytest.raises(ValueError, match="channel mismatch"):
        m(torch.randn(2, 1, 14, 64))


def test_mechanism_loss_shape_guard():
    torch = pytest.importorskip("torch")
    from src.mechanism_loss import condition_difference_loss
    with pytest.raises(ValueError, match="PAIRED"):
        condition_difference_loss(torch.randn(8, 16), torch.randn(7, 16))


# ---------------------------------------------------------------------- loader
def _write_condition(root, subjects, channels=5, T=64, labels=True,
                     cond="selfsz"):
    d = root / cond
    d.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(subjects):
        np.save(d / f"{cond}_{s}.npy",
                np.random.default_rng(i).normal(size=(channels, T))
                .astype(np.float32))
    if labels:
        lines = ["subject_id,label"] + [f"{s},{i % 2}"
                                        for i, s in enumerate(subjects)]
        (root / "labels.csv").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")


def test_label_guard_needs_ground_truth(tmp_path):
    from src.data import load_condition
    _write_condition(tmp_path, ["s01", "s02"], labels=False)
    with pytest.raises(NotImplementedError, match="ground-truth"):
        load_condition(tmp_path, "selfsz")


def test_loader_positive_path(tmp_path):
    from src.data import load_condition
    _write_condition(tmp_path, ["s01", "s02", "s03", "s04"])
    X, subj, y = load_condition(tmp_path, "selfsz",
                               labels_csv=tmp_path / "labels.csv")
    assert X.shape == (4, 5, 64)
    assert subj.tolist() == ["s01", "s02", "s03", "s04"]
    assert sorted(y.tolist()) == [0, 0, 1, 1]


def test_subject_parse_rejects_degenerate_id(tmp_path):
    """B3: 'selfsz_selfsz.npy' would collapse the whole cohort into one group."""
    from src.data import load_condition
    _write_condition(tmp_path, ["selfsz", "s02"], labels=False)
    labels = tmp_path / "labels.csv"
    labels.write_text("subject_id,label\nselfsz,1\ns02,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="degenerate subject id"):
        load_condition(tmp_path, "selfsz", labels_csv=labels)


def test_loader_requires_labels_for_every_subject(tmp_path):
    from src.data import load_condition
    _write_condition(tmp_path, ["s01", "s02", "s03", "s04"])
    labels = tmp_path / "labels.csv"
    labels.write_text("subject_id,label\ns01,1\ns02,0\n", encoding="utf-8")
    with pytest.raises(KeyError, match="missing from labels"):
        load_condition(tmp_path, "selfsz", labels_csv=labels)


def test_loader_channel_mismatch_raises(tmp_path):
    from src.data import load_condition
    _write_condition(tmp_path, ["s01", "s02"], channels=14)
    with pytest.raises(ValueError, match="expected 5 channels"):
        load_condition(tmp_path, "selfsz", labels_csv=tmp_path / "labels.csv")


def test_loader_rejects_single_class(tmp_path):
    from src.data import load_condition
    d = tmp_path / "selfsz"
    d.mkdir()
    for s in ("s01", "s02"):
        np.save(d / f"selfsz_{s}.npy", np.zeros((5, 64), dtype=np.float32))
    (tmp_path / "labels.csv").write_text("subject_id,label\ns01,1\ns02,1\n",
                                         encoding="utf-8")
    with pytest.raises(RuntimeError, match="degenerate labels"):
        load_condition(tmp_path, "selfsz", labels_csv=tmp_path / "labels.csv")


def test_loader_rejects_nan(tmp_path):
    from src.data import load_condition
    d = tmp_path / "selfsz"
    d.mkdir()
    bad = np.zeros((5, 64), dtype=np.float32)
    bad[0, 0] = np.nan
    np.save(d / "selfsz_s01.npy", bad)
    np.save(d / "selfsz_s02.npy", np.zeros((5, 64), dtype=np.float32))
    (tmp_path / "labels.csv").write_text("subject_id,label\ns01,1\ns02,0\n",
                                         encoding="utf-8")
    with pytest.raises(ValueError, match="NaN"):
        load_condition(tmp_path, "selfsz", labels_csv=tmp_path / "labels.csv")
