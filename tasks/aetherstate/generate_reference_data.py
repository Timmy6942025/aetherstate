#!/usr/bin/env python3
"""
Generate a fixed reference dataset for AetherState evolution.

This script compiles the current C++ seed engine (with the architecture from
aetherstate_bundle.py injected as #defines), runs it in generate_data mode,
and writes the resulting training records to reference_data.bin.

NOTE: This is a fast, random-weight fallback. For a stronger reference
dataset, use bootstrap_reference_data.py, which warms up a net on self-play
and then generates the final records with the trained net.  The evaluator's
auto-generation fallback now uses bootstrap_reference_data.py.

The reference dataset is used by the evaluator as a fixed, immutable training
proxy: every mutant trains on a slice of this data and is scored by validation
loss.  Because the dataset is fixed, the LLM cannot mutate the data to game
the loss metric.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from bundle_utils import (
    build_cpp_architecture_header,
    parse_bundle_architecture,
    training_record_size,
    validate_dataset,
)


def main() -> int:
    tasks_dir = Path(__file__).resolve().parent
    bundle_path = tasks_dir / "aetherstate_bundle.py"
    seed_engine_path = tasks_dir / "seed_engine.cpp"
    header_path = tasks_dir / "chess_runtime.hpp"
    output_path = tasks_dir / "reference_data.bin"

    if not seed_engine_path.is_file():
        print(f"ERROR: seed_engine.cpp not found at {seed_engine_path}", file=sys.stderr)
        return 1
    if not header_path.is_file():
        print(f"ERROR: chess_runtime.hpp not found at {header_path}", file=sys.stderr)
        return 1
    if not bundle_path.is_file():
        print(f"ERROR: {bundle_path} not found", file=sys.stderr)
        return 1

    architecture = parse_bundle_architecture(bundle_path)

    with tempfile.TemporaryDirectory(prefix="aetherstate_ref_data_") as tmpdir:
        tmp_path = Path(tmpdir)
        source_path = tmp_path / "seed_engine.cpp"
        binary_path = tmp_path / "aetherstate"
        header_dest = tmp_path / "chess_runtime.hpp"

        # Copy the required header next to the source before compiling.
        shutil.copy2(header_path, header_dest)

        # Inject architecture macros before the seed engine source.
        source_path.write_text(
            build_cpp_architecture_header(architecture) + seed_engine_path.read_text(),
            encoding="utf-8",
        )

        # Compile.
        cmd = [
            "g++",
            "-std=c++17",
            "-O3",
            str(source_path),
            "-o",
            str(binary_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            print(f"ERROR: compilation failed:\n{proc.stderr}", file=sys.stderr)
            return 1

        # Generate a fixed reference dataset.  3000 games gives a stable
        # validation-loss signal while keeping generation time reasonable.
        games = 3000
        proc = subprocess.run(
            [str(binary_path), "generate_data", str(games)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            print(f"ERROR: generate_data failed:\n{proc.stderr.decode(errors='replace')}", file=sys.stderr)
            return 1

        if not proc.stdout:
            print("ERROR: generate_data produced no output", file=sys.stderr)
            return 1

        output_path.write_bytes(proc.stdout)

        # Validate alignment and size before accepting the dataset.
        try:
            n_records = validate_dataset(output_path, architecture, min_records=10)
        except ValueError as exc:
            print(f"ERROR: dataset validation failed: {exc}", file=sys.stderr)
            return 1

        print(f"Reference dataset written to {output_path}")
        print(f"  Games: {games}")
        print(f"  Records: {n_records}")
        print(f"  Bytes: {len(proc.stdout)}")
        print(f"  Record size: {training_record_size(architecture)} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
