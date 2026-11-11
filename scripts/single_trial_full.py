#!/usr/bin/env python
"""Full-cohort single-trial classification (Barros-style, leak-free).

Scales scripts/single_trial_poc.py to all 81 subjects:
  * input: single-trial epochs, 5 midline channels (Fz FCz Cz CPz Pz),
    -100..400 ms window at 1024 Hz;
  * protocol A: subject-grouped stratified 10-fold CV (Barros 2022's
    "splitting by subject rather than by segment"), 3 seeds;
  * protocol B: leave-one-subject-out over all 81 subjects;
  * subject decision = mean trial probability, 0.5 threshold.

No scientific claim is attached to the numbers; they quantify what the
released single-trial data support under a leakage-free protocol.

Usage:
  python scripts/single_trial_full.py \
      --raw-dir D:/eegnet-data/single_trial \
      --raw-dir2 D:/eegnet-data/single_trial_ds2 --protocol 10fold
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
from sklearn.model_selection import StratifiedKFold  # noqa: E402

MIDLINE = ["Fz", "FCz", "Cz", "CPz", "Pz"]
WINDOW_MS = (-100.0, 400.0)
N_SAMPLES = 3072
T0 = time.time()


def load_time_grid() -> np.ndarray:
    t = pd.read_csv(REPO.parent / "kaggle_meta" / "time.csv", header=0)
    return t.iloc[:, 1].to_numpy(dtype=float)


def find_zip(raw_dirs, sid):
    for d in raw_dirs:
        p = Path(d) / f"{sid}.csv.zip"
        if p.exists() and p.stat().st_size > 1e6:
            return p
    return None


def load_subject(zip_path: Path, cols_hdr: list, w_idx: np.ndarray,
                 check_crc: bool = True):
    sid = int(zip_path.name.split(".")[0])
    with zipfile.ZipFile(zip_path) as zf:
        if check_crc:
            bad = zf.testzip()
            if bad:
                raise ValueError(f"corrupt zip {zip_path}: {bad}")
        name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(name) as fh:
            df = pd.read_csv(fh, header=None, names=cols_hdr, low_memory=False)
    heads = df[df["sample"] == 1][["trial", "condition"]].reset_index(drop=True)
    ch = df[MIDLINE].to_numpy(dtype=np.float32)
    if len(ch) % N_SAMPLES:
        raise ValueError(f"subject {sid}: rows not multiple of {N_SAMPLES}")
    ch = ch.reshape(-1, N_SAMPLES, len(MIDLINE))[:, w_idx, :]
    X = np.stack([ch[:, :, i] for i in range(len(MIDLINE))], axis=1)
    return sid, X, heads


def train_fold(Xtr, ytr, epochs, batch, seed, n_t):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = EEGNet(n_classes=2, n_channels=len(MIDLINE), n_timesteps=n_t)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    xt = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(1)
    yt = torch.tensor(ytr)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            lossf(model(xt[idx]), yt[idx]).backward()
            opt.step()
    return model


def eval_model(model, X, mu, sd):
    model.eval()
    with torch.no_grad():
        x = torch.tensor((X - mu) / (sd + 1e-6), dtype=torch.float32).unsqueeze(1)
        return torch.softmax(model(x), dim=1)[:, 1].numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="D:/eegnet-data/single_trial")
    ap.add_argument("--raw-dir2", default="D:/eegnet-data/single_trial_ds2")
    ap.add_argument("--protocol", choices=["10fold", "loso"], default="10fold")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--skip-crc", action="store_true",
                    help="skip zip CRC validation (archives were checked on download)")
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    labels = pd.read_csv(REPO / "data" / "labels.csv")
    diag = dict(zip(labels.subject_id.astype(int), labels.label))
    cols_hdr = pd.read_csv(REPO.parent / "kaggle_meta" / "columnLabels.csv",
                           nrows=0).columns.tolist()
    w_idx = np.where((load_time_grid() >= WINDOW_MS[0]) &
                     (load_time_grid() <= WINDOW_MS[1]))[0]
    n_t = len(w_idx)

    all_sids = sorted(diag)  # 1..81
    raw_dirs = [args.raw_dir, args.raw_dir2]
    missing = [s for s in all_sids if find_zip(raw_dirs, s) is None]
    if missing:
        print(f"MISSING {len(missing)} subjects: {missing}")
        return 1

    Xs, ys, meta = {}, {}, {}
    for sid in all_sids:
        sid, X, heads = load_subject(find_zip(raw_dirs, sid), cols_hdr, w_idx,
                                     check_crc=not args.skip_crc)
        Xs[sid], ys[sid] = X, np.full(len(X), diag[sid], dtype=np.int64)
        meta[sid] = heads
        if sid % 10 == 1:
            print(f"  loaded {sid} ({X.shape[0]} trials) {time.time()-T0:.0f}s",
                  flush=True)
    total = sum(len(v) for v in Xs.values())
    print(f"all loaded: 81 subjects, {total} trials, window {n_t} steps")

    def run_split(test_sids):
        tr = [s for s in all_sids if s not in test_sids]
        Xtr = np.concatenate([Xs[s] for s in tr])
        ytr = np.concatenate([ys[s] for s in tr])
        mu = Xtr.mean(axis=(0, 2), keepdims=True)
        sd = Xtr.std(axis=(0, 2), keepdims=True)
        Xtr_n = (Xtr - mu) / (sd + 1e-6)
        rec = []
        for seed in range(args.seeds):
            model = train_fold(Xtr_n, ytr, args.epochs, args.batch,
                               20261111 + seed, n_t)
            for s in test_sids:
                p = eval_model(model, Xs[s], mu, sd)
                rec.append({"subject": s, "diagnosis": int(ys[s][0]),
                            "seed": seed,
                            "trial_prob_mean": float(p.mean()),
                            "n_trials": int(len(p))})
        return rec

    t0 = time.time()
    if args.protocol == "10fold":
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=20261111)
        sids_arr = np.array(all_sids)
        labs_arr = np.array([diag[s] for s in all_sids])
        all_rec = []
        for k, (tr_i, te_i) in enumerate(skf.split(sids_arr, labs_arr)):
            rec = run_split([int(s) for s in sids_arr[te_i]])
            all_rec.extend(rec)
            acc = np.mean([(r["trial_prob_mean"] >= 0.5) == bool(r["diagnosis"])
                           for r in rec])
            print(f"fold {k}: {len(te_i)} subjects, fold acc {acc:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        protocol = "subject-grouped stratified 10-fold x %d seeds" % args.seeds
    else:
        all_rec = []
        for k, s in enumerate(all_sids):
            rec = run_split([s])
            all_rec.extend(rec)
            r = rec[0]
            ok = (r["trial_prob_mean"] >= 0.5) == bool(r["diagnosis"])
            print(f"loso {k+1}/81 subj {s}: prob {r['trial_prob_mean']:.3f} "
                  f"true {r['diagnosis']} {'ok' if ok else 'miss'} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        protocol = "leave-one-subject-out x %d seeds" % args.seeds

    # aggregate: per subject mean over seeds
    df = pd.DataFrame(all_rec)
    subj = df.groupby(["subject", "diagnosis"], as_index=False) \
             .agg(prob=("trial_prob_mean", "mean"), n=("n_trials", "first"))
    subj_acc = float(((subj.prob >= 0.5) == (subj.diagnosis == 1)).mean())
    pos = subj[subj.diagnosis == 1].prob.to_numpy()
    neg = subj[subj.diagnosis == 0].prob.to_numpy()
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    auc_subj = wins / (len(pos) * len(neg))
    # per-seed subject accuracy: every (subject, seed) row is one subject
    # decision obtained from the mean trial probability, so this is the same
    # quantity as subj_acc averaged with equal weight per seed.
    per_seed_acc = float((((df.trial_prob_mean >= 0.5).to_numpy()) ==
                          (df.diagnosis.to_numpy() == 1)).mean())
    out = {
        "protocol": protocol,
        "config": {"epochs": args.epochs, "batch": args.batch,
                   "channels": MIDLINE, "window_ms": WINDOW_MS,
                   "n_trials": total, "n_subjects": 81},
        "subject_level_auc": round(float(auc_subj), 3),
        "subject_level_acc": round(subj_acc, 3),
        "per_seed_subject_acc_mean": round(per_seed_acc, 3),
        "subject_decisions": subj.to_dict(orient="records"),
        "caveat": "released single-trial files carry the 64-channel montage; "
                  "the CV is grouped by subject, so trial-level leakage is "
                  "excluded. n=81 -> |d|=0.61 floor for 80% power (AUC 0.668).",
    }
    dest = REPO / f"single_trial_full_{args.protocol}.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsubject-level AUC = {auc_subj:.3f}  acc = {subj_acc:.3f} "
          f"(per-seed subject acc {per_seed_acc:.3f})")
    print(f"written {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
