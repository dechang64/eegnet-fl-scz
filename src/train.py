"""Training entrypoint (subject-based CV, leakage-safe).

AUDIT 2026-11-10
---------------
B2: fold metrics are now reported with subject groups (cluster bootstrap CI);
    aggregate uses mean/sd plus a subject-weighted pooled AUC.
B6 (med): output dir created with parents; every fold asserts zero subject
    overlap as defence in depth; positive-class index is resolved from the data
    instead of being assumed to be 1; a run manifest (git sha, config, versions,
    fold composition) is written next to the results so any number can be
    traced back to the exact state of the code that produced it.

Usage: python -m src.train --condition selfsz --model eegnet --labels labels.csv
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import load_condition
from .protocol import SubjectBasedCV, aggregate_folds, report_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name, n_channels, n_timesteps):
    from .barros_cnn import BarrosCNN
    from .eegnet import EEGNet
    if name == "eegnet":
        return EEGNet(n_channels=n_channels, n_timesteps=n_timesteps)
    if name == "barros_cnn":
        return BarrosCNN(n_channels=n_channels, n_timesteps=n_timesteps)
    raise ValueError(f"unknown model: {name}")


def to_input(x, model_name):
    t = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)
    return t.unsqueeze(1) if model_name == "eegnet" else t


def class_weights(y_train, mode, device):
    if mode != "balanced":
        return None
    counts = np.bincount(y_train, minlength=2)
    if (counts == 0).any():
        raise ValueError(f"cannot balance with empty class: counts={counts}")
    w = len(y_train) / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                                  # noqa: BLE001
        return "unknown"


def _manifest(cfg, model_name, condition, folds_subjects):
    return {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "condition": condition,
        "model": model_name,
        "config": cfg,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "fold_subject_counts": folds_subjects,
        "protocol": "subject-based StratifiedGroupKFold, per-channel z-score "
                    "fitted on the training fold only",
        "label_convention": "1 = SCZ, 0 = control",
    }


def run(condition, cfg, device, model_name="eegnet", labels_csv=None):
    X, subj, y = load_condition(Path(cfg["data_dir"]), condition,
                                labels_csv=labels_csv)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"labels must be 0/1, got {sorted(set(y.tolist()))}")
    pos = int(np.flatnonzero(np.unique(y) == 1)[0])
    print(f"[{condition}] X={X.shape}, n_subjects={len(set(subj.tolist()))}, "
          f"pos_rate={float(np.mean(y)):.3f}")

    out = Path(cfg.get("out_dir", "results"))
    out.mkdir(parents=True, exist_ok=True)
    cv = SubjectBasedCV(cfg.get("n_splits", 5), seed=cfg.get("seed", 42))

    folds, fold_subjects = [], []
    for fold, (tr, te) in enumerate(cv.split(subj, y)):
        # defence in depth: the CV already guarantees this
        if not set(subj[tr]).isdisjoint(subj[te]):
            raise RuntimeError(f"fold {fold}: subject overlap tr/te")
        set_seed(cfg.get("seed", 42) + fold)

        mu = X[tr].mean(axis=(0, 2), keepdims=True).astype(np.float32)
        sd = X[tr].std(axis=(0, 2), keepdims=True).astype(np.float32) + 1e-8
        np.savez(out / f"{condition}_{model_name}_scaler_fold{fold}.npz",
                 mu=mu, sd=sd)
        Xtr = ((X[tr] - mu) / sd).astype(np.float32)
        Xte = ((X[te] - mu) / sd).astype(np.float32)
        ytr, yte = y[tr], y[te]

        model = build_model(model_name, X.shape[1], X.shape[-1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
        w = class_weights(ytr, cfg.get("class_weight") or "none", device)
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        bs = int(cfg.get("bs", 64))
        for _ in range(int(cfg.get("epochs", 100))):
            model.train()
            idx = torch.randperm(len(Xtr)).numpy()
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                if len(b) < 2:            # BatchNorm needs >1 sample
                    continue
                xb = to_input(Xtr[b], model_name).to(device)
                yb = torch.as_tensor(ytr[b], dtype=torch.long, device=device)
                loss = lossf(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(to_input(Xte, model_name).to(device))
        if logits.shape[1] != 2:
            raise RuntimeError(f"expected 2 logits, got {logits.shape}")
        prob = logits.softmax(1)[:, pos].cpu().numpy()
        pred = logits.argmax(1).cpu().numpy()
        m = report_metrics(yte, pred, prob, groups=subj[te])
        m["fold"] = fold
        m["n_subjects_test"] = int(len(set(subj[te].tolist())))
        m["n_subjects_train"] = int(len(set(subj[tr].tolist())))
        folds.append(m)
        fold_subjects.append(m["n_subjects_test"])
        print(f"  fold {fold}: acc={m['acc']:.3f} auc={m['auc']:.3f} "
              f"sens={m['sens']:.3f} spec={m['spec']:.3f} "
              f"auc95={m['auc_ci'][0]:.3f}-{m['auc_ci'][1]:.3f} "
              f"(n_test_subj={m['n_subjects_test']})")
        torch.save(model.state_dict(),
                   out / f"{condition}_{model_name}_fold{fold}.pt")

    agg = aggregate_folds(folds, fold_subjects)
    payload = _manifest(cfg, model_name, condition, fold_subjects)
    payload.update(per_fold=folds, aggregate_mean_std=agg)
    (out / f"{condition}_{model_name}_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{condition}|{model_name}]",
          {k: f"{v[0]:.3f}+/-{v[1]:.3f}" for k, v in agg.items()})
    return agg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--condition", default="selfsz",
                   choices=["selfsz", "passivesz", "buttonsz"])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default="eegnet",
                   choices=["eegnet", "barros_cnn"])
    p.add_argument("--labels", default=None,
                   help="CSV with header subject_id,label (1=SCZ, 0=control)")
    p.add_argument("--class-weight", default="none",
                   choices=["none", "balanced"])
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    if a.class_weight != "none":
        cfg["class_weight"] = a.class_weight
    run(a.condition, cfg, a.device, a.model, labels_csv=a.labels)
