#!/usr/bin/env python3
"""Safely align the pipeline's publication top-k SD threshold with the manuscript.

The final revision uses 0.01. This helper creates a backup, changes exactly one
matching configuration assignment if needed, and refuses ambiguous edits.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


PATTERN = re.compile(
    r"(stability_topk_std_threshold\s*(?::\s*float\s*)?=\s*)"
    r"(0\.0?5|0\.01)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("src/small_data_mof_benchmark_pipeline.py"),
    )
    args = parser.parse_args()
    path = args.path
    if not path.exists():
        raise SystemExit(f"ERROR: {path} does not exist.")

    text = path.read_text(encoding="utf-8")
    matches = list(PATTERN.finditer(text))
    if len(matches) != 1:
        raise SystemExit(
            f"ERROR: expected exactly one stability_topk_std_threshold assignment; "
            f"found {len(matches)}. Edit manually and rerun repository validation."
        )

    old = matches[0].group(2)
    if old == "0.01":
        print(f"OK: {path} already uses stability_topk_std_threshold = 0.01")
        return

    backup = path.with_suffix(path.suffix + ".before_publication_threshold.bak")
    shutil.copy2(path, backup)
    new_text = PATTERN.sub(r"\g<1>0.01", text, count=1)
    path.write_text(new_text, encoding="utf-8")
    print(f"UPDATED: {path}")
    print(f"BACKUP : {backup}")
    print("Changed stability_topk_std_threshold from 0.05 to 0.01.")


if __name__ == "__main__":
    main()
