"""The provenance checker must fail loudly, or it is worse than nothing.

A verifier that passes on any input would let a corrupted or truncated
download be cited as the official release. These tests build fake copies and
assert the script refuses them.
"""
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_hashes.py"
sys.path.insert(0, str(ROOT))

from scripts.check_hashes import EXPECTED  # noqa: E402


def run_check(directory):
    return subprocess.run([sys.executable, str(SCRIPT), "--dir", str(directory)],
                          capture_output=True, text=True)


def test_digests_are_well_formed():
    for name, (digest, size) in EXPECTED.items():
        assert name.endswith(".csv")
        assert len(digest) == 64 and int(digest, 16) >= 0
        assert size > 0


def test_wrong_content_is_rejected(tmp_path):
    # Same name and roughly the same size, different bytes.
    bad = tmp_path / "ERPdata.csv"
    bad.write_bytes(b"subject,condition\n1,1\n")
    r = run_check(tmp_path)
    assert r.returncode == 1, r.stdout
    assert "MISMATCH" in r.stdout


def test_truncated_file_is_rejected(tmp_path):
    name = "ERPdata.csv"
    want_size = EXPECTED[name][1]
    f = tmp_path / name
    f.write_bytes(b"\0" * min(want_size, 4096))
    r = run_check(tmp_path)
    assert r.returncode == 1, r.stdout
    assert "MISMATCH" in r.stdout


def test_size_match_alone_is_not_enough(tmp_path):
    """Right length, wrong bytes: size is a weak check and must not pass."""
    name = "demographic.csv"
    size = EXPECTED[name][1]
    (tmp_path / name).write_bytes(b"x" * size)
    r = run_check(tmp_path)
    assert r.returncode == 1
    assert hashlib.sha256(b"x" * size).hexdigest() != EXPECTED[name][0]


def test_missing_files_do_not_fail_the_run(tmp_path):
    """A partial download should report, not crash - and not claim success."""
    r = run_check(tmp_path)
    assert r.returncode == 0, r.stdout
    assert "absent" in r.stdout
    assert "byte-identical" not in r.stdout


def test_empty_directory_is_explicit(tmp_path):
    r = run_check(tmp_path)
    assert "Nothing to compare" in r.stdout


def test_bad_directory_exits_nonzero(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "--dir",
                        str(tmp_path / "nope")], capture_output=True, text=True)
    assert r.returncode != 0
    assert "not a directory" in (r.stdout + r.stderr)


def test_zip_member_is_verified_by_content(tmp_path):
    """Kaggle ships some files inside a zip; hashing must follow the member."""
    import zipfile

    name = "demographic.csv"
    payload = b"subject,diagnosis\n1,SZ\n"
    zp = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, payload)
    r = run_check(tmp_path)
    # Payload is not the official file, so it must be reported as a mismatch
    # (found and hashed) rather than silently skipped as absent.
    assert "MISMATCH" in r.stdout, r.stdout
    assert f"{name}.zip" in r.stdout
