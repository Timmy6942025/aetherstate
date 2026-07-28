#!/usr/bin/env python3
"""
Rebuild aetherstate_bundle.py from the current standalone source files.

Usage:
    python build_bundle.py

This reads:
  - tasks/aetherstate/seed_engine.cpp -> SEED_ENGINE_CPP
  - tasks/aetherstate/train_loop.py  -> TRAIN_LOOP_PY

And writes them into tasks/aetherstate/aetherstate_bundle.py while preserving
its ARCHITECTURE dict and header comments.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    tasks_dir = Path(__file__).resolve().parent
    bundle_path = tasks_dir / "aetherstate_bundle.py"
    seed_engine_path = tasks_dir / "seed_engine.cpp"
    train_loop_path = tasks_dir / "train_loop.py"

    if not bundle_path.is_file():
        print(f"ERROR: bundle not found at {bundle_path}", file=sys.stderr)
        return 1
    if not seed_engine_path.is_file():
        print(f"ERROR: seed_engine.cpp not found at {seed_engine_path}", file=sys.stderr)
        return 1
    if not train_loop_path.is_file():
        print(f"ERROR: train_loop.py not found at {train_loop_path}", file=sys.stderr)
        return 1

    bundle = bundle_path.read_text(encoding="utf-8")
    seed_engine = seed_engine_path.read_text(encoding="utf-8")
    train_loop = train_loop_path.read_text(encoding="utf-8")

    def _replace_block(src: str, marker: str, replacement: str) -> str:
        """Replace the raw triple-quoted block after `marker` with `replacement`."""
        start = src.find(marker)
        if start == -1:
            raise ValueError(f"Could not locate marker: {marker}")
        # Find the closing triple-quote that belongs to this block.
        # The marker line is immediately followed by a newline and the block content.
        content_start = start + len(marker)
        if src[content_start] != "\n":
            raise ValueError(f"Expected newline after marker: {marker}")
        content_start += 1
        end = src.find("\n'''", content_start)
        if end == -1:
            raise ValueError(f"Could not find closing triple-quote for marker: {marker}")
        # end is the start of the newline before the closing triple-quote.
        return src[:content_start] + replacement + "\n'''" + src[end + 4 :]

    bundle = _replace_block(bundle, "SEED_ENGINE_CPP = r'''", seed_engine)
    bundle = _replace_block(bundle, "TRAIN_LOOP_PY = r'''", train_loop)

    bundle_path.write_text(bundle, encoding="utf-8")
    print(f"Rebuilt {bundle_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
