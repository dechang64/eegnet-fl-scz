#!/usr/bin/env python
"""Single-trial proof of concept on the 6-subject 64-channel subset.

Barros-style pipeline, scaled honestly down:
  * input: single-trial epochs, 5 midline channels (Fz FCz Cz CPz Pz),
    -100..400 ms window at 1024 Hz -> 512 timesteps;
  * model: the repo's EEGNet, unmodified;
  * protocol: leave-one-subject-out (6 folds). No trial of the held-out
    subject ever touches training or normalization statistics;
  * subject decision = mean of that subject's trial probabilities, 0.5
    threshold (as in Barros 2022).

With n=6 this makes no scientific claim. It demonstrates that the
mechanism-probe pipeline (single trials in, subject-level aggregation
out) runs end to end on the real 64-channel files and quantifies how
far trial-level and subject-level metrics diverge under leakage-free CV.

Usage:
  python scripts/single_trial_poc.py --raw-dir D:/eegnet-data/single_trial
"""
import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from src.eegnet import EEGNet  # noqa: E402

MIDLINE = ["Fz", "FCz", "Cz", "CPz", "Pz"]
WINDOW_MS = (-100.0, 400.0)
N_SAMPLES = 3072
SEED = 20261111


def load_time_grid() -> np.ndarray:
    t = pd.read_csv(REPO.parent / "kaggle_meta" / "time.csv", header=0)
    return t.iloc[:, 1].to_numpy(dtype=float)


def load_subject(zip_path: Path, cols_hdr: list, w_idx: np.ndarray) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(name) as fh:
            df = pd.read_csv(fh, header=None, names=cols_hdr, low_memory=False)
    # one row per (trial, condition) block head; keep index columns
    heads = df[df["sample"] == 1][["trial", "condition"]]
    ch = df[MIDLINE].to_numpy(dtype=np.float32)          # (rows, 5)
    ch = ch.reshape(-1, N_SAMPLES, len(MIDLINE))[:, w_idx, :]  # (trials, T, 5)
    out = heads.reset_index(drop=True)
    for i, c in enumerate(MIDLINE):
        out["x_" + c] = list(ch[:, :, i])                # (T,) arrays
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="D:/eegnet-data/single_trial")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads() // 2))

    labels = pd.read_csv(REPO / "data" / "labels.csv")
    diag = dict(zip(labels.subject_id.astype(int), labels.label))

    cols_hdr = pd.read_csv(REPO.parent / "kaggle_meta" / "columnLabels.csv",
                           nrows=0).columns.tolist()
    time_ms = load_time_grid()
    w_idx = np.where((time_ms >= WINDOW_MS[0]) & (time_ms <= WINDOW_MS[1]))[0]
    n_t = len(w_idx)
    subjects = sorted(int(p.name.split(".")[0]) for p in
                      Path(args.raw_dir).glob("*.csv.zip"))
    print(f"subjects: {subjects}  window: {n_t} timesteps")

    data = {}
    for sid in subjects:
        df = load_subject(Path(args.raw_dir) / f"{sid}.csv.zip", cols_hdr, w_idx)
        X = np.stack([np.stack(df["x_" + c].to_numpy()) for c in MIDLINE],
                     axis=1)                              # (trials, 5, T)
        y = np.full(len(df), diag[sid], dtype=np.int64)
        data[sid] = (X, y, df["condition"].to_numpy())
        print(f"  subject {sid}: {X.shape[0]} trials, label={diag[sid]}")

    results = []
    for sid in subjects:
        t0 = time.time()
        tr_sids = [s for s in subjects if s != sid]
        Xtr = np.concatenate([data[s][0] for s in tr_sids])
        ytr = np.concatenate([data[s][1] for s in tr_sids])
        Xte, yte, _ = data[sid]

        # normalization from training subjects only (leak-free)
        mu = Xtr.mean(axis=(0, 2), keepdims=True)        # per channel over trials & time
        sd = Xtr.std(axis=(0, 2), keepdims=True)
        Xtr_n = (Xtr - mu) / (sd + 1e-6)
        Xte_n = (Xte - mu) / (sd + 1e-6)

        model = EEGNet(n_classes=2, n_channels=len(MIDLINE), n_timesteps=n_t)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        xt = torch.tensor(Xtr_n, dtype=torch.float32).unsqueeze(1)  # (N,1,C,T)
        yt = torch.tensor(ytr)
        g = torch.Generator().manual_seed(SEED)
        model.train()
        for ep in range(args.epochs):
            perm = torch.randperm(len(xt), generator=g)
            for i in range(0, len(xt), args.batch):
                idx = perm[i:i + args.batch]
                opt.zero_grad()
                loss = lossf(model(xt[idx]), yt[idx])
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            xe = torch.tensor(Xte_n, dtype=torch.float32).unsqueeze(1)
            prob = torch.softmax(model(xe), dim=1)[:, 1].numpy()
        subj_prob = float(prob.mean())
        subj_pred = int(subj_prob >= 0.5)
        results.append({
            "subject": sid,
            "diagnosis": int(yte[0]),
            "n_trials": int(len(prob)),
            "trial_prob_mean": subj_prob,
            "subject_pred_correct": int(subj_pred == int(yte[0])),
            "seconds": round(time.time() - t0, 1),
        })
        print(f"  fold {sid}: mean prob {subj_prob:.3f} "
              f"(true {yte[0]}) {results[-1]['seconds']}s")

    # subject-level: 6 points, AUC by rank of mean prob
    probs = np.array([r["trial_prob_mean"] for r in results])
    ys = np.array([r["diagnosis"] for r in results])
    pos, neg = probs[ys == 1], probs[ys == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    auc_subj = wins / (len(pos) * len(neg))
    acc_subj = float(np.mean([(r["trial_prob_mean"] >= 0.5) == bool(r["diagnosis"])
                              for r in results]))

    out = {
        "protocol": "LOSO 6 folds, EEGNet, 5 midline ch, -100..400 ms @1024Hz",
        "config": {"subjects": subjects, "epochs": args.epochs,
                   "batch": args.batch, "seed": SEED},
        "folds": results,
        "subject_level_auc_6pts": round(float(auc_subj), 3),
        "subject_level_acc_6pts": acc_subj,
        "caveat": "n=6; pipeline demonstration only, no scientific claim",
    }
    dest = REPO / "single_trial_poc.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsubject-level AUC(6 pts) = {auc_subj:.3f}  acc = {acc_subj:.2f}")
    print(f"written {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
