"""Generate a synthetic button-tone-sz-shaped dataset for pipeline smoke tests.

This is NOT data and must never be reported as a result. It exists so the
loader -> CV -> model -> metrics -> XAI path can be executed end-to-end on a
machine without Kaggle access, which is the only way to know the code runs
before real data arrives.

The generator encodes the hypothesis under test: controls show an attenuated
N100 for self-generated tones, patients show a reduced suppression (Ford &
Mathalon 2004; Barros 2022). If the pipeline cannot recover a signal it was
built to contain, it is broken.

Usage:
  python scripts/make_synthetic.py --out data_synth --subjects 24
  python -m src.train --condition selfsz --model eegnet \
      --config configs/default.yaml --labels data_synth/labels.csv
  (set data_dir in the config, or use --config after editing)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CHANNELS, META_NAME       # noqa: E402

N_CH = len(CHANNELS)  # 9 documented sites - must match src/data.py exactly
FS = 500.0
T = 320               # 640 ms epochs, as in the button-tone design

# Relative N100 amplitude per site, in CHANNELS order
# (Fz FCz Cz FC3 C3 CP3 FC4 C4 CP4): fronto-central maximum, decaying
# laterally and posteriorly. Shape only - it is not a fitted topography.
TOPO = np.array([1.0, 0.90, 0.60, 0.70, 0.45, 0.30, 0.70, 0.45, 0.30])


def n100_epoch(rng, suppressed: float, base_amp: float = 6.0):
    """One epoch: background 1/f-ish noise + a negative deflection at ~100 ms
    whose amplitude is multiplied by `suppressed` (1.0 = full N100, small =
    suppressed)."""
    t = np.arange(T) / FS
    noise = rng.normal(0, 0.6, size=(N_CH, T))
    # 1/f shaping via cumulative sum, then re-standardise
    noise = np.cumsum(noise, axis=1)
    noise -= noise.mean(axis=1, keepdims=True)
    noise /= noise.std(axis=1, keepdims=True) + 1e-9
    latency, width = 0.100, 0.025
    n100 = -base_amp * np.exp(-0.5 * ((t - latency) / width) ** 2)
    return (noise * 1.0 + suppressed * np.outer(TOPO, n100)).astype(np.float32)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data_synth")
    p.add_argument("--subjects", type=int, default=24)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    rows = ["subject_id,label"]
    half = a.subjects // 2
    for i in range(a.subjects):
        label = 1 if i < half else 0            # 1 = SCZ
        sid = f"s{i + 1:03d}"
        rows.append(f"{sid},{label}")
        # controls: strong suppression (self-generated sound is predicted, N100
        # attenuated); patients: weak suppression (corollary discharge deficit)
        base_suppression = 0.35 if label == 0 else 0.85
        for cond in ("selfsz", "passivesz", "buttonsz"):
            d = out / cond
            d.mkdir(parents=True, exist_ok=True)
            if cond == "selfsz":
                damp = base_suppression + rng.normal(0, 0.05)
            elif cond == "buttonsz":
                damp = min(1.0, base_suppression + 0.1 + rng.normal(0, 0.05))
            else:                                # passive: N100 not attenuated
                damp = 1.0 + rng.normal(0, 0.03)
            arr = np.stack([n100_epoch(rng, damp) for _ in range(a.epochs)])
            np.save(d / f"{cond}_{sid}.npy", arr)

    (out / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"synthetic dataset written to {out}/ "
          f"({a.subjects} subjects, {a.epochs} epochs x 3 conditions)")
    print("NOTE: synthetic - any AUC produced from it validates the pipeline, "
          "not the hypothesis.")


if __name__ == "__main__":
    main()
