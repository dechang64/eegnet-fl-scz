"""Resumable downloader for the per-subject single-trial CSVs of Barros et al. 2022.

The two Kaggle mirrors partition the 81 subjects:
  * broach/button-tone-sz   -> subjects 1-24 and 66-81 (40 subjects)
  * broach/buttontonesz2    -> subjects 25-65        (41 subjects)

Each subject is a single ~200 MB zipped CSV. Downloads are done with plain
HTTPS against the Kaggle API so that a single file can be fetched without
pulling the whole mirror (the full dataset is ~9 GB).

Usage:
    KAGGLE_KEY=... python scripts/download_single_trial.py --dest-root D:/eegnet-data

The script is idempotent: an existing archive is kept only if the central
directory and every member CRC validate, otherwise it is re-downloaded.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

DATASET_OF = {}
for _s in list(range(1, 25)) + list(range(66, 82)):
    DATASET_OF[_s] = "broach/button-tone-sz"
for _s in range(25, 66):
    DATASET_OF[_s] = "broach/buttontonesz2"

SUBDIR_OF = {
    "broach/button-tone-sz": "single_trial",
    "broach/buttontonesz2": "single_trial_ds2",
}

API = "https://www.kaggle.com/api/v1/datasets/download/{dataset}/{filename}"
USER = "dechang64"


def archive_ok(path: Path) -> bool:
    """True only if the zip opens, its directory is intact and all CRCs pass."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            return bad is None and len(zf.namelist()) > 0
    except (zipfile.BadZipFile, OSError):
        return False


def download(url: str, dest: Path, key: str, chunk: int = 1 << 20) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"Authorization": f"Bearer {key}"}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        last = 0.0
        with open(tmp, "wb") as fh:
            for block in r.iter_content(chunk_size=chunk):
                if not block:
                    continue
                fh.write(block)
                done += len(block)
                now = time.time()
                if now - last > 30:
                    last = now
                    pct = f"{100 * done / total:5.1f}%" if total else "  ?  "
                    print(f"    {pct} {done / 1e6:7.1f} MB", flush=True)
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest-root", type=Path, default=Path("D:/eegnet-data"))
    ap.add_argument("--subjects", default="", help="comma list, default = all 81")
    ap.add_argument("--max-tries", type=int, default=3)
    args = ap.parse_args()

    key = os.environ.get("KAGGLE_KEY", "").strip()
    if not key:
        print("KAGGLE_KEY is not set", file=sys.stderr)
        return 2

    subjects = (
        [int(x) for x in args.subjects.split(",") if x.strip()]
        if args.subjects
        else sorted(DATASET_OF)
    )

    failures = []
    for i, subj in enumerate(subjects, 1):
        dataset = DATASET_OF[subj]
        dest_dir = args.dest_root / SUBDIR_OF[dataset]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{subj}.csv.zip"

        if archive_ok(dest):
            print(f"[{i:2d}/{len(subjects)}] {subj:3d} ok (cached, "
                  f"{dest.stat().st_size / 1e6:.0f} MB)", flush=True)
            continue

        if dest.exists():
            print(f"[{i:2d}/{len(subjects)}] {subj:3d} cached archive invalid, "
                  f"re-downloading", flush=True)
            dest.unlink()

        for attempt in range(1, args.max_tries + 1):
            try:
                print(f"[{i:2d}/{len(subjects)}] {subj:3d} <- {dataset}", flush=True)
                download(API.format(dataset=dataset, filename=f"{subj}.csv"), dest, key)
                if not archive_ok(dest):
                    raise RuntimeError("downloaded archive failed CRC check")
                print(f"[{i:2d}/{len(subjects)}] {subj:3d} done "
                      f"({dest.stat().st_size / 1e6:.0f} MB)", flush=True)
                break
            except Exception as exc:  # noqa: BLE001 - network is unreliable
                print(f"    attempt {attempt} failed: {exc}", flush=True)
                if attempt == args.max_tries:
                    failures.append((subj, str(exc)))
                else:
                    time.sleep(5 * attempt)

    print(f"\nfinished: {len(subjects) - len(failures)}/{len(subjects)} ok")
    for subj, err in failures:
        print(f"  FAILED {subj}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
