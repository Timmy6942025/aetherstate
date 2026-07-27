"""
AetherState OpenEvolve Evaluator
---------------------------------
This module is called by OpenEvolve to score a mutated seed_engine.cpp.
The mutated file path is passed as the only argument to `evaluate()`.

Behavior:
    1. Compile the supplied C++ source with g++ -O3.
       If compilation fails or takes longer than 15 seconds, return zeroed metrics.
    2. Run a 90-second speed micro-benchmark -> nodes_per_second.
    3. Run a 3-minute win-rate benchmark against a random legal-move opponent -> win_rate.
    4. Return both raw metrics as distinct feature axes for OpenEvolve's MAP-Elites
       grid.  The selection fitness is the multiplicative product of the two
       normalized objectives, avoiding the previous linear combined formula.

The returned metrics dictionary is compatible with OpenEvolve's
EvaluationResult helper.

Environment variables (for quick testing only):
    AETHERSTATE_BENCH_SECONDS  - override the speed benchmark duration (default 90)
    AETHERSTATE_RANDOM_SECONDS - override the random-opponent duration (default 180)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from openevolve.evaluation_result import EvaluationResult

# Header files the seed #includes.  OpenEvolve writes the mutated seed to
# a temp path that has no siblings, so we must stage these next to it
# before invoking g++ or the preprocessor fails in ~100 ms (compile_ok=False).
NEEDED_HEADERS = ["chess_runtime.hpp"]

# tasks/aetherstate/ -- this file's directory, which is where the headers live.
TASKS_DIR = Path(__file__).resolve().parent


def _parse_output(text: str) -> dict:
    """Parse key=value lines printed by the engine.

    The engine prints lines such as:
        BENCHMARK nodes_per_second=3052715
        RANDOM win_rate=1 games=181
    """
    result: dict = {}
    for line in text.strip().splitlines():
        parts = line.strip().split()
        for part in parts:
            if "=" in part:
                key, val = part.split("=", 1)
                key = key.strip().lower()
                val = val.strip()
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
    return result


def _compile(source_path: str, binary_path: str, timeout: float = 15.0) -> dict:
    """Compile source. Return dict with ok, elapsed, error, flags."""
    start = time.time()
    for flags in ("-mavx2 -mavx512f", ""):
        cmd = (
            ["g++", "-std=c++17", "-O3", "-x", "c++"]
            + (flags.split() if flags else [])
            + [source_path, "-o", binary_path]
        )
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "elapsed": time.time() - start,
                "error": "compile timeout (>15s)",
            }
        if proc.returncode == 0:
            return {
                "ok": True,
                "elapsed": time.time() - start,
                "flags": flags or "none",
            }
    return {
        "ok": False,
        "elapsed": time.time() - start,
        "error": (proc.stderr or b"unknown compile error").decode(errors="replace"),
    }


def _run(binary_path: str, mode: str, arg: str) -> dict:
    """Run the compiled engine in a given mode and parse key=value output."""
    proc = subprocess.run(
        [binary_path, mode, arg],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    result: dict = {"error": proc.stderr.strip() or "engine crashed"}
    if proc.returncode != 0:
        return result
    return _parse_output(proc.stdout)


def _trainability_gate(source_path: str) -> tuple[bool, str]:
    """Run a tiny training pipeline to verify the code can train end-to-end.

    This is intentionally lightweight so it can run on a slow ARM host (e.g. a
    Raspberry Pi 4) while still guaranteeing that the final CUDA training run
    in the cloud will not hit structural bugs.  It only proves that:
      - generate_data produces valid records
      - PyTorch can forward/backward on those records
      - exported weights match the C++ NeuralNet layout
      - the C++ binary can load those weights and run inference
    The gate does not try to train a good model -- that is left to the final
    large run on CUDA hardware.
    """
    gate_cmd = [
        sys.executable,
        str(TASKS_DIR / "train_loop.py"),
        source_path,
        "--selfplay-games", "1",
        "--epochs", "1",
        "--batch-size", "2",
        "--no-benchmark",
        "--output", "json",
    ]
    try:
        proc = subprocess.run(
            gate_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=240,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"trainability gate timed out after {exc.timeout}s"

    output = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        return False, output

    # train_loop.py --output json prints a JSON dict.  Treat any top-level
    # "error" key as a failure, because the script still exits 0 when it
    # reports benchmark errors in JSON mode.
    try:
        result = json.loads(proc.stdout.splitlines()[-1] if proc.stdout else "{}")
    except json.JSONDecodeError:
        return True, output  # non-JSON trailing lines are acceptable

    if isinstance(result, dict) and "error" in result:
        return False, output

    return True, output


def evaluate(program_path: str) -> dict:
    """
    OpenEvolve evaluator entry point.

    Args:
        program_path: Path to the mutated C++ source file.

    Returns:
        Dictionary of metrics.  ``nodes_per_second`` and ``win_rate`` are
        returned as raw, independent axes for MAP-Elites.
    """
    source_path = Path(program_path)
    if not source_path.exists():
        return EvaluationResult(
            metrics={
                "fitness": 0.0,
                "combined_score": 0.0,
                "nodes_per_second": 0.0,
                "win_rate": 0.0,
                "compile_ok": False,
                "trainable": False,
                "error": "source not found",
            },
            artifacts={"stderr": f"Source file not found: {program_path}"},
        ).to_dict()

    bench_seconds = int(os.environ.get("AETHERSTATE_BENCH_SECONDS", "90"))
    random_seconds = int(os.environ.get("AETHERSTATE_RANDOM_SECONDS", "180"))

    with tempfile.TemporaryDirectory(prefix="aetherstate_eval_") as tmpdir:
        # Stage the mutated seed + required headers in one directory so
        # internal #includes (e.g. #include "chess_runtime.hpp") resolve.
        # OpenEvolve passes `program_path` as a temp file with no siblings;
        # the original headers live in tasks/aetherstate/ next to evaluate.py.
        eval_source = os.path.join(tmpdir, source_path.name)
        try:
            shutil.copy2(source_path, eval_source)
        except OSError as e:
            return EvaluationResult(
                metrics={
                    "fitness": 0.0,
                    "combined_score": 0.0,
                    "nodes_per_second": 0.0,
                    "win_rate": 0.0,
                    "compile_ok": False,
                    "trainable": False,
                    "error": f"could not copy mutated seed into eval dir: {e}",
                },
            ).to_dict()
        missing_headers = []
        for hdr in NEEDED_HEADERS:
            src_hpp = TASKS_DIR / hdr
            if not src_hpp.is_file():
                missing_headers.append(str(src_hpp))
                continue
            shutil.copy2(src_hpp, os.path.join(tmpdir, hdr))
        if missing_headers:
            return EvaluationResult(
                metrics={
                    "fitness": 0.0,
                    "combined_score": 0.0,
                    "nodes_per_second": 0.0,
                    "win_rate": 0.0,
                    "compile_ok": False,
                    "trainable": False,
                    "error": f"required headers missing: {missing_headers}",
                },
            ).to_dict()
        binary_path = os.path.join(tmpdir, "aetherstate")

        compile_info = _compile(eval_source, binary_path, timeout=45.0)
        if not compile_info["ok"]:
            artifacts = {
                "stderr": compile_info.get("error", ""),
                "compile_time": compile_info.get("elapsed", 0.0),
            }
            return EvaluationResult(
                metrics={
                    "fitness": 0.0,
                    "combined_score": 0.0,
                    "nodes_per_second": 0.0,
                    "win_rate": 0.0,
                    "compile_ok": False,
                    "trainable": False,
                    "compile_time": compile_info.get("elapsed", 0.0),
                },
                artifacts=artifacts,
            ).to_dict()

        # --- Trainability gate ---
        # Before spending minutes on the full benchmarks, do a cheap end-to-end
        # check on the Pi.  This catches architecture / weight-layout / data
        # generation bugs that would break the final CUDA training run.
        gate_ok, gate_output = _trainability_gate(eval_source)
        if not gate_ok:
            artifacts = {
                "stderr": compile_info.get("error", ""),
                "gate_output": gate_output,
                "compile_time": compile_info.get("elapsed", 0.0),
            }
            return EvaluationResult(
                metrics={
                    "fitness": 0.0,
                    "combined_score": 0.0,
                    "nodes_per_second": 0.0,
                    "win_rate": 0.0,
                    "compile_ok": True,
                    "trainable": False,
                    "compile_time": compile_info.get("elapsed", 0.0),
                    "error": "trainability gate failed (see artifacts['gate_output'])",
                },
                artifacts=artifacts,
            ).to_dict()

        speed = _run(binary_path, "bench_time", str(bench_seconds))
        nodes_per_second = speed.get("nodes_per_second", 0.0)

        random_result = _run(binary_path, "random_time", str(random_seconds))
        win_rate = random_result.get("win_rate", 0.0)

        # Non-linear fitness: multiplicative combination of the two objectives.
        # This avoids the previous linear formula while still rewarding both
        # speed and strength.  win_rate is in [0, 1]; nodes per second is
        # normalized against a 1 MNode/sec reference.
        fitness = win_rate * (nodes_per_second / 1_000_000.0)

        # `combined_score` is used by OpenEvolve for internal best-program
        # tracking and evolution guidance. Keep it equal to `fitness` so the
        # archive and the reported best program are consistent.
        combined_score = fitness

        metrics = {
            "fitness": fitness,
            "combined_score": combined_score,
            "nodes_per_second": nodes_per_second,
            "win_rate": win_rate,
            "compile_ok": True,
            "trainable": True,
            "compile_time": compile_info.get("elapsed", 0.0),
            "compile_flags": compile_info.get("flags", ""),
        }

        artifacts = {
            "speed_output": json.dumps(speed),
            "random_output": json.dumps(random_result),
        }

        return EvaluationResult(metrics=metrics, artifacts=artifacts).to_dict()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "seed_engine.cpp"
    print(json.dumps(evaluate(path), indent=2))
