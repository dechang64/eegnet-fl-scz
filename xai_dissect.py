"""XAI dissection: does a label-only-trained EEGNet independently rediscover
the N100 corollary-discharge signature?

AUDIT 2026-11-10
---------------
B7 (med): the backward pass hard-coded logit index 1 as "SCZ". The label
    convention is fixed by the labels CSV, so the index is now resolved from
    the data and can be overridden; a wrong index silently inverts every
    topographic conclusion.
B7b: gradient figures previously averaged over every epoch in the condition,
    including subjects the model was trained on. That answers "what does the
    model encode", not "what generalises". `--holdout-subjects` restricts the
    analysis to a fold's held-out subjects and the choice is printed.
B4b: the montage dict only knew 5 midline channels; a different montage now
    raises instead of plotting wrong positions. It is now `CHANNEL_POS`, covering
    the nine sites the dataset actually provides.
B15: the virtual-ERP time axis started at 0 while button-tone-sz epochs are
    centred on the stimulus, so the N100 window was shaded ~1.5 s off the true
    latency - a figure that looks right and is wrong. Latency is now measured
    from the event, taken from dataset_meta.json.
B14: `FS` was a module constant guessed at 500 Hz and every time axis was
    `np.arange(T)/FS`. A wrong constant shifts the N100 band on every figure
    while plotting cleanly. fs now comes from `dataset_meta.json`, which the
    ingester derives from the real time column.

Usage:
  python xai_dissect.py --condition selfsz --labels labels.csv \
      --checkpoint results/selfsz_eegnet_fold4.pt \
      --scaler results/selfsz_eegnet_scaler_fold4.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import torch                             # noqa: E402

from src.data import (CHANNEL_POS, CHANNELS, load_condition,   # noqa: E402
                      resolve_event_onset, resolve_fs)
from src.eegnet import EEGNet                                    # noqa: E402

N100_WINDOW = (0.080, 0.150)


def load_model(ckpt, n_channels, n_timesteps):
    model = EEGNet(n_channels=n_channels, n_timesteps=n_timesteps)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def temporal_kernel_fft(model, out_dir: Path, fs: float):
    w = model.block1[0].weight.data.cpu().numpy()
    kern = w.shape[-1]
    freqs = np.fft.rfftfreq(kern, d=1.0 / fs)
    fig, axes = plt.subplots(4, 2, figsize=(10, 9), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        if i >= w.shape[0]:
            ax.axis("off")
            continue
        spec = np.abs(np.fft.rfft(w[i, 0, 0] - w[i, 0, 0].mean()))
        ax.plot(freqs, spec, lw=1.4)
        ax.axvspan(4, 30, color="orange", alpha=0.15)
        ax.set_title(f"temporal kernel {i}", fontsize=9)
    axes[-1, 0].set_xlabel("Hz")
    axes[-1, 1].set_xlabel("Hz")
    fig.suptitle("Temporal kernels - frequency response")
    fig.tight_layout()
    fig.savefig(out_dir / "temporal_kernels_fft.png", dpi=200)
    plt.close(fig)
    print("saved temporal_kernels_fft.png")


def spatial_weights(model, out_dir: Path):
    n_ch = model.n_channels
    if n_ch != len(CHANNELS):
        raise ValueError(
            f"model has {n_ch} channels but CHANNEL_POS describes "
            f"{len(CHANNELS)} ({CHANNELS}); extend CHANNEL_POS before plotting")
    w = model.depthwise.weight.data.cpu().numpy()       # (F1*D, 1, C, 1)
    per_ch = np.linalg.norm(w[:, 0, :, 0], axis=0)
    per_ch = per_ch / (per_ch.sum() + 1e-12)
    xs = [CHANNEL_POS[c][0] for c in CHANNELS]
    ys = [CHANNEL_POS[c][1] for c in CHANNELS]
    arc = np.linspace(-0.15, 0.15, 50)
    fig, ax = plt.subplots(figsize=(4, 6))
    sc = ax.scatter(xs, ys, s=300 + 4000 * per_ch, c=per_ch,
                    cmap="viridis", zorder=3)
    ax.plot(arc, np.sqrt(np.maximum(0.72 ** 2 - arc ** 2, 0)), color="0.5")
    ax.plot(arc, -np.sqrt(np.maximum(0.72 ** 2 - arc ** 2, 0)), color="0.5")
    for c, x, yv in zip(CHANNELS, xs, ys):
        ax.annotate(c, (x, yv), xytext=(6, 4), textcoords="offset points")
    ax.set_title("Depthwise spatial weights")
    ax.axis("off")
    fig.colorbar(sc, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / "spatial_weights.png", dpi=200)
    plt.close(fig)
    print("saved spatial_weights.png")


def virtual_erp(model, X_std, out_dir: Path, fs: float, onset_s: float = 0.0,
                class_index: int = 1):
    """Per-sample |d(logit_class)/dinput|, averaged.

    Per-sample backward, not sum-then-backward: signed gradients cancel when
    summed, which erased the very temporal structure being looked for (H2).
    """
    if X_std.ndim != 3:
        raise ValueError(f"expected (N, C, T), got {X_std.shape}")
    G = np.zeros_like(X_std, dtype=np.float32)
    for i in range(len(X_std)):
        xi = torch.as_tensor(X_std[i:i + 1]).unsqueeze(1).clone()
        xi.requires_grad_(True)
        model(xi)[0, class_index].backward()
        G[i] = xi.grad[0, 0].detach().abs().numpy()
    g = G.mean(axis=(0, 1))
    # Latency must be measured from the EVENT, not from the first sample: the
    # epochs are centred on the stimulus, so sample 0 is ~1.5 s before it.
    t = np.arange(len(g)) / fs - onset_s
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, g, lw=1.5)
    ax.axvspan(*N100_WINDOW, color="red", alpha=0.15, label="N100 window")
    ax.axvline(0.0, color="0.4", lw=0.8, ls="--", label="event")
    ax.set_xlabel("time from event (s)")
    ax.set_ylabel("mean |grad|")
    ax.set_title(f"Virtual ERP (class {class_index} logit)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "virtual_erp.png", dpi=200)
    plt.close(fig)
    print("saved virtual_erp.png")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--condition", default="selfsz")
    p.add_argument("--labels", required=True,
                   help="CSV with header subject_id,label (1=SCZ, 0=control)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scaler", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="xai_results")
    p.add_argument("--class-index", type=int, default=1,
                   help="logit index of SCZ; 1 when labels are 1=SCZ")
    p.add_argument("--holdout-subjects", default=None,
                   help="comma-separated subject ids to analyse (a fold's "
                        "held-out set). Omit to use all subjects.")
    a = p.parse_args(argv)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X, subj, y = load_condition(Path(a.data_dir), a.condition,
                                labels_csv=a.labels)
    if a.class_index not in (0, 1):
        raise ValueError("--class-index must be 0 or 1 for binary EEGNet")

    sel = np.ones(len(subj), dtype=bool)
    if a.holdout_subjects:
        wanted = {s.strip() for s in a.holdout_subjects.split(",") if s.strip()}
        unknown = wanted - set(subj.tolist())
        if unknown:
            raise KeyError(f"unknown subjects requested: {sorted(unknown)}")
        sel = np.isin(subj, list(wanted))
        print(f"[XAI] held-out subset: {len(wanted)} subjects, "
              f"{int(sel.sum())} epochs")
    else:
        print("[XAI] WARNING: analysing all subjects (including training "
              "subjects); pass --holdout-subjects for a generalisation claim")

    fs = resolve_fs(a.data_dir)
    onset = resolve_event_onset(a.data_dir)
    print(f"[XAI] fs={fs:g} Hz, epoch={X.shape[-1]} samples = "
          f"{X.shape[-1] / fs:.3f} s, event at {onset:.3f} s "
          f"(from {a.data_dir}/dataset_meta.json)")
    if not (0.0 <= onset < X.shape[-1] / fs):
        raise ValueError(
            f"event onset {onset:.3f}s lies outside the epoch "
            f"(0..{X.shape[-1] / fs:.3f}s)")
    sc = np.load(a.scaler)
    X_std = ((X[sel] - sc["mu"]) / sc["sd"]).astype(np.float32)
    model = load_model(a.checkpoint, X.shape[1], X.shape[-1])
    temporal_kernel_fft(model, out_dir, fs)
    spatial_weights(model, out_dir)
    virtual_erp(model, X_std[:64], out_dir, fs, onset_s=onset,
                class_index=a.class_index)
    print(f"XAI dissection complete -> {out_dir}")


if __name__ == "__main__":
    main()
