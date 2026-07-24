#!/usr/bin/env python3
"""Insert a Zenodo DOI into README, data-availability documentation, and CFF."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def normalize_doi(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^https?://doi\.org/", "", value, flags=re.I)
    if not re.fullmatch(r"10\.\d{4,9}/\S+", value):
        raise ValueError(f"Not a valid-looking DOI: {value}")
    return value


def replace_block(path: Path, doi: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"<!-- ZENODO_DOI_START -->.*?<!-- ZENODO_DOI_END -->",
        flags=re.S,
    )
    replacement = (
        "<!-- ZENODO_DOI_START -->\n"
        f"Archived software release: https://doi.org/{doi}\n"
        "<!-- ZENODO_DOI_END -->"
    )
    new_text, n = pattern.subn(replacement, text)
    if n != 1:
        raise RuntimeError(f"Expected one DOI marker block in {path}; found {n}.")
    path.write_text(new_text, encoding="utf-8")


def update_cff(path: Path, doi: str) -> None:
    text = path.read_text(encoding="utf-8")
    if re.search(r"(?m)^doi:\s*", text):
        text = re.sub(r'(?m)^doi:\s*.*$', f'doi: "{doi}"', text, count=1)
    else:
        marker = "# SOFTWARE_DOI_INSERTION_POINT"
        if marker not in text:
            raise RuntimeError(f"Missing insertion marker in {path}.")
        text = text.replace(marker, f'doi: "{doi}"\n{marker}')
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doi", required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    doi = normalize_doi(args.doi)
    root = args.root.resolve()

    replace_block(root / "README.md", doi)
    replace_block(root / "docs" / "DATA_AVAILABILITY.md", doi)
    update_cff(root / "CITATION.cff", doi)
    print(f"Inserted DOI {doi} into README.md, DATA_AVAILABILITY.md, and CITATION.cff")


if __name__ == "__main__":
    main()
