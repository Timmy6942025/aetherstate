#!/usr/bin/env python3
"""
Bootstrap a stronger reference dataset for AetherState.

This script improves the fixed reference dataset quality without requiring
external engines or PGN files.  It:

  1. Compiles the current C++ engine with the architecture from
     aetherstate_bundle.py.
  2. Generates a small "warm-up" dataset using the untrained (random-weight)
     net.
  3. Trains the PyTorch model on that warm-up data for a few epochs.
  4. Exports trained int8 weights.
  5. Generates the final reference dataset by loading the trained weights
     into the C++ engine and letting it play self-play games.

Because the trained net encodes some chess patterns from the warm-up self-play,
the resulting reference games are stronger and more structured than random play,
giving the evolution evaluator a better val_loss signal.

Usage:
    python bootstrap_reference_data.py [options]

Options:
    --warm-games <N>      Games for the initial warm-up dataset (default 200)
    --final-games <N>     Games for the final reference dataset (default 1000)
    --warm-epochs <N>     Epochs to train on warm-up data (default 3)
    --batch-size <N>      Training batch size (default 256)

The script writes its output to tasks/aetherstate/reference_data.bin.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from bundle_utils import (
    build_cpp_architecture_header,
    build_python_arch_constants,
    parse_bundle_architecture,
    training_record_size,
    validate_dataset,
)


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, **kwargs)


def generate_with_binary(binary_path: Path, games: int) -> bytes:
    proc = _run(
        [str(binary_path), "generate_data", str(games)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"generate_data failed:\n{proc.stderr.decode(errors='replace')}")
    return proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap a stronger AetherState reference dataset")
    parser.add_argument("--warm-games", type=int, default=200, help="Warm-up games (default 200)")
    parser.add_argument("--final-games", type=int, default=3000, help="Final reference games (default 3000)")
    parser.add_argument("--warm-epochs", type=int, default=3, help="Epochs on warm-up data (default 3)")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size (default 256)")
    args = parser.parse_args()

    tasks_dir = Path(__file__).resolve().parent
    bundle_path = tasks_dir / "aetherstate_bundle.py"
    seed_engine_path = tasks_dir / "seed_engine.cpp"
    header_path = tasks_dir / "chess_runtime.hpp"
    train_loop_path = tasks_dir / "train_loop.py"
    output_path = tasks_dir / "reference_data.bin"

    if not bundle_path.is_file():
        print(f"ERROR: {bundle_path} not found", file=sys.stderr)
        return 1
    if not seed_engine_path.is_file():
        print(f"ERROR: seed_engine.cpp not found at {seed_engine_path}", file=sys.stderr)
        return 1
    if not header_path.is_file():
        print(f"ERROR: chess_runtime.hpp not found at {header_path}", file=sys.stderr)
        return 1
    if not train_loop_path.is_file():
        print(f"ERROR: train_loop.py not found at {train_loop_path}", file=sys.stderr)
        return 1

    architecture = parse_bundle_architecture(bundle_path)

    with tempfile.TemporaryDirectory(prefix="aetherstate_ref_data_") as tmpdir:
        tmp_path = Path(tmpdir)
        source_path = tmp_path / "seed_engine.cpp"
        binary_path = tmp_path / "aetherstate"
        weights_path = tmp_path / "weights.bin"
        header_dest = tmp_path / "chess_runtime.hpp"
        train_dest = tmp_path / "train_loop.py"

        # Stage source files in the temp directory.
        shutil.copy2(header_path, header_dest)
        shutil.copy2(train_loop_path, train_dest)
        source_path.write_text(
            build_cpp_architecture_header(architecture) + seed_engine_path.read_text(),
            encoding="utf-8",
        )

        # Compile.
        compile_cmd = ["g++", "-std=c++17", "-O3", str(source_path), "-o", str(binary_path)]
        proc = _run(compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            print(f"ERROR: compilation failed:\n{proc.stderr}", file=sys.stderr)
            return 1

        # 1) Generate warm-up data with random weights.
        print("Generating warm-up self-play data (random weights)...")
        warm_data = generate_with_binary(binary_path, args.warm_games)
        warm_path = tmp_path / "warm_data.bin"
        warm_path.write_bytes(warm_data)
        try:
            warm_records = validate_dataset(warm_path, architecture, min_records=10)
        except ValueError as exc:
            print(f"ERROR: warm-up dataset validation failed: {exc}", file=sys.stderr)
            return 1
        print(f"  Warm-up records: {warm_records}")

        # 2) Train a PyTorch model on the warm-up data.
        #    Write arch_constants.py next to the temp train_loop.py so the
        #    training loop uses the bundle architecture instead of legacy defaults.
        (tmp_path / "arch_constants.py").write_text(
            build_python_arch_constants(architecture), encoding="utf-8"
        )

        print("Training PyTorch model on warm-up data...")
        train_cmd = [
            sys.executable,
            str(train_dest),
            str(source_path),
            "--dataset", str(warm_path),
            "--epochs", str(args.warm_epochs),
            "--batch-size", str(args.batch_size),
            "--no-benchmark",
            "--weights-path", str(weights_path),
            "--output", "json",
        ]
        train_proc = _run(
            train_cmd,
            cwd=str(tmp_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600.0,
        )
        if train_proc.returncode != 0:
            print(f"ERROR: warm-up training failed:\n{train_proc.stderr}", file=sys.stderr)
            return 1
        print(f"  Training output: {train_proc.stdout.strip().splitlines()[-1]}")
        if not weights_path.is_file():
            print("ERROR: weights were not exported", file=sys.stderr)
            return 1

        # 3) Generate final reference data with trained weights.
        print("Generating final reference data with trained weights...")
        proc = _run(
            [str(binary_path), "load_weights", str(weights_path), "generate_data", str(args.final_games)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            print(f"ERROR: final generate_data failed:\n{proc.stderr.decode(errors='replace')}", file=sys.stderr)
            return 1

        if not proc.stdout:
            print("ERROR: final generate_data produced no output", file=sys.stderr)
            return 1

        # Write to a temporary file and atomically rename. This prevents a
        # partial/corrupted reference_data.bin if the process is interrupted,
        # and makes concurrent fallback generation safe (each writer produces a
        # complete file; the last atomic rename wins).
        tmp_output = output_path.with_name(f"reference_data_tmp_{os.getpid()}.bin")
        tmp_output.write_bytes(proc.stdout)
        try:
            n_records = validate_dataset(tmp_output, architecture, min_records=10)
        except ValueError as exc:
            tmp_output.unlink(missing_ok=True)
            print(f"ERROR: dataset validation failed: {exc}", file=sys.stderr)
            return 1
        tmp_output.replace(output_path)

        print(f"  Final reference dataset written to {output_path}")
        print(f"  Games: {args.final_games}")
        print(f"  Records: {n_records}")
        print(f"  Bytes: {len(proc.stdout)}")
        print(f"  Record size: {training_record_size(architecture)} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
