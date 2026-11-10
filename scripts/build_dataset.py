"""Convert the Kaggle export into the layout src/data.py expects.

Every mapping is explicit. If a role, an electrode or a diagnosis value cannot
be resolved unambiguously, this script raises instead of picking something
plausible - see AUDIT.md.

Usage:
  python scripts/build_dataset.py --erp ERPdata.csv \
      --demographics demographic.csv --out data
  # 81 averaged ERPs/condition -> one epoch per subject per condition.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ingest import (IngestError, detect_column, read_demographics,  # noqa: E402
                        stack_long, stack_wide, write_dataset)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--erp", required=True, help="ERPdata.csv")
    p.add_argument("--demographics", required=True, help="demographic.csv")
    p.add_argument("--out", default="data")
    p.add_argument("--subject-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--condition-col", default=None)
    p.add_argument("--electrode-col", default=None)
    p.add_argument("--time-col", default=None)
    p.add_argument("--value-col", default=None)
    p.add_argument("--scz-values", default=None,
                   help="comma-separated diagnosis values meaning patient")
    p.add_argument("--condition-map", default=None,
                   help='JSON, e.g. \'{"selfsz":"self"}\'')
    a = p.parse_args(argv)

    df = pd.read_csv(a.erp, encoding="utf-8-sig")
    cols = {
        "subject": a.subject_col or detect_column(df, "subject"),
        "condition": a.condition_col or detect_column(df, "condition"),
        "electrode": a.electrode_col or detect_column(df, "electrode"),
        "time": a.time_col or detect_column(df, "time"),
        "value": a.value_col or detect_column(df, "value"),
    }
    for role in ("subject", "condition", "time"):
        if cols[role] is None:
            raise SystemExit(
                f"could not resolve the {role!r} column in {a.erp}; columns: "
                f"{list(df.columns)}. Pass --{role}-col.")

    overrides = json.loads(a.condition_map) if a.condition_map else None
    if cols["electrode"]:
        epochs = stack_long(df, cols, overrides)
        layout = "long"
    else:
        epochs = stack_wide(df, cols, overrides)
        layout = "wide"

    scz = ([s.strip() for s in a.scz_values.split(",")]
           if a.scz_values else None)
    labels = read_demographics(a.demographics, a.subject_col, a.label_col, scz)

    result = write_dataset(
        epochs, labels, a.out,
        meta_extra={"source_csv": Path(a.erp).name,
                    "source_demographics": Path(a.demographics).name,
                    "layout": layout,
                    "synthetic": False})
    meta = result["meta"]
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nwrote {result['written_files']} parquet files + labels.csv + "
          f"dataset_meta.json to {a.out}/")
    print("NOTE: ERPdata.csv holds TRIAL-AVERAGED waveforms, so each subject "
          "contributes 1 sample per condition. Report that number, not just "
          "the AUC.")


if __name__ == "__main__":
    try:
        main()
    except IngestError as exc:
        raise SystemExit(f"ingest refused: {exc}") from exc
