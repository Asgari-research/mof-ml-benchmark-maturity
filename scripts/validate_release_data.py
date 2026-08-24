#!/usr/bin/env python3
"""Validate the processed benchmark input distributed with the publication release."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

EXPECTED = {
    "filename",
    "Crystalnet",
    "uptake(mmol/g) CO2 at 0.015 bar",
    "uptake(mmol/g) CO2 at 0.15 bar",
    "uptake(mmol/g) methane at 5.8 bar",
    "uptake(mmol/g) methane at 65 bar",
}

def sha256_stream(fh):
    h = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
        h.update(chunk)
        size += len(chunk)
    return h.hexdigest(), size

def main():
    root = Path(__file__).resolve().parents[1]
    archive = root / "data" / "clean_data.zip"
    manifest_path = root / "data" / "clean_data_manifest.json"

    if not archive.exists():
        raise RuntimeError("Missing data/clean_data.zip")
    if not manifest_path.exists():
        raise RuntimeError("Missing data/clean_data_manifest.json")

    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if archive_sha != manifest["archive_sha256"]:
        raise RuntimeError("data/clean_data.zip checksum does not match manifest")

    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
        members = [n for n in zf.namelist() if not n.endswith("/") and Path(n).name.lower() == "clean_data.csv"]
        if len(members) != 1:
            raise RuntimeError(f"Expected exactly one clean_data.csv; found {members}")
        member = members[0]
        with zf.open(member) as fh:
            member_sha, member_size = sha256_stream(fh)
        if member_sha != manifest["member_sha256"]:
            raise RuntimeError("clean_data.csv checksum does not match manifest")
        with zf.open(member) as fh:
            reader = csv.reader(io.TextIOWrapper(fh, encoding="utf-8-sig", newline=""))
            header = next(reader)
            rows = sum(1 for _ in reader)

    missing = sorted(EXPECTED - set(header))
    if missing:
        raise RuntimeError(f"Missing expected columns: {missing}")
    if rows != int(manifest["data_rows_excluding_header"]):
        raise RuntimeError("CSV row count does not match manifest")
    if len(header) != int(manifest["column_count"]):
        raise RuntimeError("CSV column count does not match manifest")

    print("PASS: processed benchmark release data")
    print("rows:", rows)
    print("columns:", len(header))
    print("archive_sha256:", archive_sha)
    print("member_sha256:", member_sha)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
