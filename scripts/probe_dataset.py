"""Report the real structure of the Kaggle export. Read-only, no assumptions.

Run this FIRST, before building anything. It only inspects and prints; it
writes nothing to the dataset. The report is what tells you which flags
build_dataset.py needs.

Usage:
  python scripts/probe_dataset.py --erp ERPdata.csv --demographics demographic.csv
  python scripts/probe_dataset.py --erp ERPdata.csv --json d1_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CHANNELS, CONDITIONS          # noqa: E402
from src.ingest import (IngestError, canonical_condition,  # noqa: E402
                        detect_column, infer_fs, read_demographics)


def probe(erp_path: Path, demographics: Path | None, condition_map=None) -> dict:
    df = pd.read_csv(erp_path, encoding="utf-8-sig")
    rep: dict = {
        "file": erp_path.name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "n_missing_cells": int(df.isna().sum().sum()),
    }
    roles = {}
    for role in ("subject", "condition", "electrode", "time", "value"):
        try:
            roles[role] = detect_column(df, role)
        except IngestError as exc:
            roles[role] = None
            rep.setdefault("warnings", []).append(str(exc))
    rep["roles"] = roles
    rep["layout"] = "long" if roles["electrode"] else "wide"

    for role in ("subject", "time"):
        if roles[role] is None:
            rep.setdefault("warnings", []).append(
                f"could not resolve {role!r}; columns are {list(df.columns)}")
    if rep.get("warnings") and (roles["subject"] is None or roles["time"] is None):
        return rep

    t = pd.to_numeric(df[roles["time"]], errors="coerce").dropna()
    try:
        fs, unit = infer_fs(t)
        rep["fs"] = fs
        rep["time_unit"] = unit
        rep["n_samples_per_series"] = int(t.nunique())
        rep["epoch_seconds"] = rep["n_samples_per_series"] / fs
        rep["time_range"] = [float(t.min()), float(t.max())]
    except IngestError as exc:
        rep.setdefault("warnings", []).append(f"fs not inferable: {exc}")

    if roles["condition"] is not None:
        raw_tokens = sorted({str(v) for v in df[roles["condition"]].unique()})
        rep["condition_tokens_raw"] = raw_tokens
        mapped = {}
        for tok in raw_tokens:
            try:
                mapped[tok] = canonical_condition(tok, condition_map)
            except IngestError:
                mapped[tok] = "UNMAPPED"
        rep["condition_tokens_mapped"] = mapped
        unmapped = [k for k, v in mapped.items() if v == "UNMAPPED"]
        if unmapped:
            rep.setdefault("warnings", []).append(
                "unmapped condition tokens " + str(unmapped)
                + " - pass --condition-map to name them")

    subjects = sorted({str(v).strip() for v in df[roles["subject"]].unique()})
    rep["n_subjects"] = len(subjects)
    rep["subject_examples"] = subjects[:8]

    if roles["electrode"] is not None:
        elecs = sorted({str(v) for v in df[roles["electrode"]].unique()})
        rep["n_electrodes"] = len(elecs)
        rep["electrodes"] = elecs
        rep["expected_9_sites_present"] = [
            c for c in CHANNELS if c.lower() in {e.lower() for e in elecs}]
        rep["expected_9_sites_missing"] = [
            c for c in CHANNELS if c.lower() not in {e.lower() for e in elecs}]

    if roles["condition"] is not None:
        per = {}
        for tok, g in df.groupby(roles["condition"]):
            per[str(tok)] = {
                "rows": int(len(g)),
                "subjects": int(g[roles["subject"]].nunique()),
                "electrodes": (int(g[roles["electrode"]].nunique())
                               if roles["electrode"] else None),
            }
        rep["per_condition"] = per
        cov = (df.groupby([roles["subject"], roles["condition"]]).size()
               .unstack(fill_value=0))
        incomplete = [str(i) for i in cov.index if (cov.loc[i] == 0).any()]
        rep["subjects_missing_a_condition"] = incomplete

    if demographics is not None:
        try:
            labels = read_demographics(demographics)
            rep["labels"] = {
                "file": demographics.name,
                "n_labelled": len(labels),
                "n_scz": int(sum(labels.values())),
                "n_control": int(len(labels) - sum(labels.values())),
                "subjects_in_eeg_without_label":
                    [s for s in subjects if s not in labels],
            }
        except (IngestError, Exception) as exc:      # noqa: BLE001
            rep["labels"] = {"error": str(exc)}
    return rep


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--erp", required=True, help="ERPdata.csv (or a superset)")
    p.add_argument("--demographics", default=None)
    p.add_argument("--condition-map", default=None,
                   help='JSON, e.g. \'{"selfsz":"self","passivesz":"passive"}\'')
    p.add_argument("--json", default=None, help="write the report here")
    a = p.parse_args(argv)

    cmap = json.loads(a.condition_map) if a.condition_map else None
    rep = probe(Path(a.erp),
                Path(a.demographics) if a.demographics else None, cmap)
    text = json.dumps(rep, indent=2, ensure_ascii=False)
    if a.json:
        Path(a.json).write_text(text, encoding="utf-8")
    print(text)
    print("\n--- next step ---")
    if rep.get("warnings"):
        print("resolve the warnings above, then rerun. Do not build yet.")
    else:
        print("python scripts/build_dataset.py --erp %s --demographics %s "
              "--out data" % (a.erp, a.demographics or "<demographic.csv>"))


if __name__ == "__main__":
    main()
