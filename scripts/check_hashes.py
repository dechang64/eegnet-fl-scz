"""Verify that a local copy of button-tone-sz matches the official Kaggle release.

Why this exists: the D1 run was first executed on a third-party Zenodo copy
(record 19104585), which left an honest caveat on every number. On 2026-11-10
the official files were downloaded from Kaggle and all three turned out
byte-identical, so the caveat can be dropped - but it should stay checkable,
not just asserted in prose.

Expected digests are of the *decompressed* CSVs as served by
    /api/v1/datasets/download/broach/button-tone-sz/<file>
Note that Kaggle serves ERPdata.csv and mergedTrialData.csv inside a zip but
demographic.csv as a bare file, so this script accepts either form.

Usage:
    python scripts/check_hashes.py --dir kaggle_raw
    python scripts/check_hashes.py --dir data_raw --json provenance.json

Exit code 0 only when every file present matches; missing files are reported
but do not fail, so a partial download still gives a useful answer.
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

# sha256 of the official release, verified 2026-11-10 against
# kaggle.com/datasets/broach/button-tone-sz (version 6)
EXPECTED = {
    "ERPdata.csv": ("16c34c14cfe5dccccf9ecf1dd4d7e0bd4bc513e0849b6866d3f61037951bad33",
                    120_228_250),
    "mergedTrialData.csv": ("3da285b58bcdb8559040b3f1bff32f2a8d96b2f6ccda5f1156f9fe6bca326f36",
                            15_361_541),
    "demographic.csv": ("417d655ab85343b84c77ffef33c7a3d2835df6233bacedc4034bda808e6c1416",
                        1_566),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_stream(fh) -> str:
    h = hashlib.sha256()
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
    return h.hexdigest()


def candidate_paths(root: Path, name: str):
    """Bare file, or <name>.zip / <name> inside a zip Kaggle produced."""
    yield root / name
    for zp in (root / f"{name}.zip", root / name.replace(".csv", ".zip")):
        if zp.exists():
            try:
                with zipfile.ZipFile(zp) as zf:
                    for member in zf.namelist():
                        if member.endswith(name):
                            yield zp, member
            except zipfile.BadZipFile:
                # Kaggle sometimes serves the CSV with a .zip-sounding URL but
                # no compression; treat it as a bare file.
                yield zp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="kaggle_raw")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    root = Path(a.dir)
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")

    report, failures, missing = {}, [], []
    for name, (want_hash, want_size) in EXPECTED.items():
        found = False
        for cand in candidate_paths(root, name):
            if isinstance(cand, tuple):
                zp, member = cand
                with zipfile.ZipFile(zp) as zf:
                    with zf.open(member) as fh:
                        got_hash = sha256_stream(fh)
                    got_size = zf.getinfo(member).file_size
                source = f"{zp.name}:{member}"
            else:
                if not cand.exists():
                    continue
                got_hash, got_size = sha256_file(cand), cand.stat().st_size
                source = cand.name
            found = True
            ok = (got_hash == want_hash and got_size == want_size)
            report[name] = {"source": source, "sha256": got_hash,
                            "size": got_size, "matches_official": ok}
            if ok:
                print(f"OK      {name:<20} {got_size:>12,} bytes  {got_hash[:16]}...  ({source})")
            else:
                failures.append(name)
                print(f"MISMATCH {name:<20} {got_size:>12,} bytes  {got_hash[:16]}...  ({source})")
                print(f"         expected {want_size:>12,} bytes  {want_hash[:16]}...")
            break
        if not found:
            missing.append(name)
            print(f"absent  {name:<20} (not found under {root})")

    print()
    if failures:
        print(f"{len(failures)} file(s) DIFFER from the official release: {failures}")
        print("Do not cite numbers computed from them without explaining the difference.")
    elif missing and not report:
        print("Nothing to compare.")
    elif missing:
        print(f"{len(report)} file(s) verified identical; {len(missing)} absent: {missing}")
    else:
        print("All files byte-identical to the official Kaggle release.")

    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {a.json}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
