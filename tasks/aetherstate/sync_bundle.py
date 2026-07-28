#!/usr/bin/env python3
"""Sync the standalone seed_engine.cpp and train_loop.py into aetherstate_bundle.py.

OpenEvolve mutates aetherstate_bundle.py, but the standalone files are easier to
edit and test.  After changing the standalone files, run this script so the
bundle reflects the latest code.
"""

import ast
import sys
from pathlib import Path


TARGETS = {
    "SEED_ENGINE_CPP": "seed_engine.cpp",
    "TRAIN_LOOP_PY": "train_loop.py",
}


def _find_assignment_end(source: str, start_idx: int) -> int:
    """Return the index after the closing triple-quote of a raw string assignment."""
    # The assignment looks like:
    #   IDENT = r'''...'''\n
    # Find the opening r'''
    quote_start = source.find("r'''", start_idx)
    if quote_start == -1:
        raise ValueError("Could not find opening r''' after assignment start")
    # The actual content starts right after r'''
    content_start = quote_start + 4
    # Find the closing '''
    close_quote = source.find("'''", content_start)
    if close_quote == -1:
        raise ValueError("Could not find closing ''' for raw string assignment")
    return close_quote + 3


def sync_bundle(bundle_path: Path) -> None:
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)

    tasks_dir = bundle_path.parent
    source = bundle_path.read_text(encoding="utf-8")

    for target_name, file_name in TARGETS.items():
        standalone_path = tasks_dir / file_name
        if not standalone_path.is_file():
            print(f"Warning: {standalone_path} not found, skipping", file=sys.stderr)
            continue

        new_content = standalone_path.read_text(encoding="utf-8")

        # Locate the assignment: NAME = r'''
        assign_str = f"{target_name} = r'''"
        start_idx = source.find(assign_str)
        if start_idx == -1:
            raise ValueError(f"Could not find {target_name} assignment in {bundle_path}")

        content_start = start_idx + len(assign_str)
        # The closing triple quote is the first "'''" that is NOT preceded by a
        # backslash and appears at the start of a line (after optional whitespace).
        close_pos = content_start
        while True:
            close_pos = source.find("'''", close_pos)
            if close_pos == -1:
                raise ValueError(f"Could not find closing ''' for {target_name}")
            # Check that it is at the start of a line (allowing leading whitespace)
            line_start = source.rfind("\n", content_start, close_pos) + 1
            leading = source[line_start:close_pos]
            if leading.strip() == "":
                break
            close_pos += 3

        # Build the replacement
        before = source[:start_idx]
        after = source[close_pos + 3 :]
        source = f"{before}{target_name} = r\'\'\'{new_content}\'\'\'{after}"
        print(f"Synced {target_name} from {standalone_path}")

    bundle_path.write_text(source, encoding="utf-8")
    print(f"Updated {bundle_path}")


if __name__ == "__main__":
    bundle_path = Path(__file__).resolve().parent / "aetherstate_bundle.py"
    sync_bundle(bundle_path)
