"""
AetherState OpenEvolve Evaluator (bundle / trainability gate)
-------------------------------------------------------------
This evaluator is called by OpenEvolve with a single argument: the path to an
``aetherstate_bundle.py`` file.  The bundle contains:
  - ARCHITECTURE dict (single source of truth for dimensions)
  - SEED_ENGINE_CPP string
  - TRAIN_LOOP_PY string

The evaluator:
  1. Parses the bundle.
  2. Validates the architecture against safe bounds.
  3. Injects the architecture into the C++ and Python sources.
  4. Compiles, generates data, micro-trains, exports/loads weights, and
     benchmarks inference speed.
  5. Returns trainability metrics.  It does NOT run real training or chess
     games during evolution.

Environment variables (for quick testing only):
    AETHERSTATE_BENCH_SECONDS            - speed-benchmark duration in seconds (default 2)
    AETHERSTATE_RANDOM_SECONDS           - random-opponent match duration in seconds (default 5)
    AETHERSTATE_SELFPLAY_GAMES           - self-play games for micro-training (default 64)
    AETHERSTATE_NUM_SEEDS                - number of random seeds for Tier 1 (default 1)
    AETHERSTATE_PER_SEED_TIMEOUT         - max seconds per seed run (default 180)
    AETHERSTATE_TIER2_WIN_RATE_THRESHOLD - win-rate gate for Tier 2 validation (default 0.6)
    AETHERSTATE_TIER2_BENCH_SECONDS      - Tier 2 speed-benchmark duration (default 10)
    AETHERSTATE_TIER2_RANDOM_SECONDS     - Tier 2 random-opponent duration (default 30)
    AETHERSTATE_TIER2_SELFPLAY_GAMES     - Tier 2 self-play games (default 256)
    AETHERSTATE_TIER2_TIMEOUT            - max seconds for Tier 2 (default 600)
    AETHERSTATE_TIER2_REPLACES_FITNESS   - if "1", use Tier 2 metrics for fitness (default 0)
    AETHERSTATE_RUN_TIER2                - if "0", skip Tier 2 entirely (default 1)
"""

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from openevolve.evaluation_result import EvaluationResult

# Load the sibling bundle_utils module by absolute path so this evaluator works
# whether it is run as a script or imported dynamically by OpenEvolve.
_BUNDLE_UTILS_PATH = Path(__file__).resolve().parent / "bundle_utils.py"
_bundle_utils_spec = importlib.util.spec_from_file_location(
    "aetherstate_bundle_utils", str(_BUNDLE_UTILS_PATH)
)
_bundle_utils = importlib.util.module_from_spec(_bundle_utils_spec)
_bundle_utils_spec.loader.exec_module(_bundle_utils)
REQUIRED_ARCHITECTURE_KEYS = _bundle_utils.REQUIRED_ARCHITECTURE_KEYS

# Header files the seed #includes.  OpenEvolve writes the mutated bundle to a
# temp path with no siblings, so we must stage these next to it before invoking
# g++ or the preprocessor fails.
NEEDED_HEADERS = ["chess_runtime.hpp"]

# tasks/aetherstate/ -- this file's directory, which is where the headers live.
TASKS_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Architecture guardrails
# ---------------------------------------------------------------------------
ARCHITECTURE_LIMITS = {
    "input_features": (768, 768),     # fixed by bitboard encoding
    "output_slots": (4096, 4096),     # fixed by move encoding
    "move_stride": (64, 64),          # must match output_slots encoding
    "max_features_per_record": (8, 128),
    "weight_version": (1, 65535),     # versioned weight format for topology experiments
    # Mutable topology keys: wide bounds allow architectural exploration while
    # preventing pathological OOM/slow-compile mutations.
    "accumulator_size": (16, 4096),
    "hidden_size": (8, 1024),
    "quant_shift": (1, 15),
}


# Weight file magic must never change; the evaluator rejects mutations to it.
REQUIRED_WEIGHT_MAGIC = "AESTATEW"



def _parse_bundle(program_path: str) -> dict:
    """Parse an aetherstate_bundle.py file into its components using AST only.

    Returns a dict with keys: architecture, seed_engine_cpp, train_loop_py.
    Raises ValueError if the bundle is malformed.

    We deliberately do NOT execute the bundle; it is LLM-generated code and must
    be treated as untrusted text until it is validated.
    """
    path = Path(program_path)
    if not path.exists():
        raise ValueError(f"Bundle not found: {program_path}")

    source = path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"Bundle has a Python syntax error: {e}") from e

    namespace: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        namespace[target.id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass

    architecture = namespace.get("ARCHITECTURE")
    seed_engine_cpp = namespace.get("SEED_ENGINE_CPP")
    train_loop_py = namespace.get("TRAIN_LOOP_PY")
    research_notebook = namespace.get("RESEARCH_NOTEBOOK")

    if not isinstance(architecture, dict):
        raise ValueError("Bundle must define an ARCHITECTURE dict")
    if not isinstance(seed_engine_cpp, str):
        raise ValueError("Bundle must define a SEED_ENGINE_CPP string")
    if not isinstance(train_loop_py, str):
        raise ValueError("Bundle must define a TRAIN_LOOP_PY string")

    return {
        "architecture": architecture,
        "seed_engine_cpp": seed_engine_cpp,
        "train_loop_py": train_loop_py,
        "research_notebook": research_notebook if isinstance(research_notebook, str) else "",
    }


def _validate_architecture(architecture: dict) -> tuple[bool, str]:
    """Validate the architecture dict against guardrails.

    Returns (ok, error_message).
    """
    missing = REQUIRED_ARCHITECTURE_KEYS - set(architecture.keys())
    if missing:
        return False, f"Missing required architecture keys: {sorted(missing)}"

    for key, (min_val, max_val) in ARCHITECTURE_LIMITS.items():
        value = architecture.get(key)
        if not isinstance(value, int):
            return False, f"Architecture key '{key}' must be an integer, got {type(value).__name__}"
        if value < min_val or value > max_val:
            return False, (
                f"Architecture key '{key}'={value} is outside safe bounds "
                f"[{min_val}, {max_val}]"
            )

    if architecture.get("weight_magic") != REQUIRED_WEIGHT_MAGIC:
        return False, (
            f"Architecture weight_magic must remain '{REQUIRED_WEIGHT_MAGIC}'. "
            "Do not change the weight file format."
        )

    if architecture.get("output_slots") != architecture.get("move_stride", 0) * 64:
        return False, (
            "output_slots must equal move_stride * 64 (fixed by from/to move encoding)"
        )

    if not isinstance(architecture.get("weight_version"), int):
        return False, "weight_version must be an integer"

    return True, ""


def _build_arch_constants_py(architecture: dict) -> str:
    """Generate the Python arch_constants.py module from the architecture dict."""
    input_features = architecture["input_features"]
    accumulator_size = architecture["accumulator_size"]
    hidden_size = architecture["hidden_size"]
    output_slots = architecture["output_slots"]
    weight_magic = architecture["weight_magic"]
    max_features = architecture["max_features_per_record"]

    return f"""# Auto-generated by AetherState evaluator.  Do not edit by hand.
INPUT_FEATURES = {input_features}
ACCUMULATOR_SIZE = {accumulator_size}
HIDDEN_SIZE = {hidden_size}
OUTPUT_SLOTS = {output_slots}
WEIGHT_MAGIC = {weight_magic!r}.encode("ascii")
# NET_SIZE intentionally removed: topology is mutable. Python exporter and C++ loader must agree.

import numpy as np
TRAINING_RECORD_DTYPE = np.dtype([
    ("side_to_move", "<i4"),
    ("n_features", "<i4"),
    ("features", "<i4", {max_features}),
    ("from_", "<i4"),
    ("to_", "<i4"),
    ("promo", "<i4"),
    ("outcome", "<i4"),
])
"""


def _build_cpp_architecture_header(architecture: dict) -> str:
    """Generate the C++ architecture #define block prepended to seed_engine.cpp."""
    return f"""// Auto-generated by AetherState evaluator.  Do not edit by hand.
#define INPUT_FEATURES {architecture['input_features']}
#define ACCUMULATOR_SIZE {architecture['accumulator_size']}
#define HIDDEN_SIZE {architecture['hidden_size']}
#define OUTPUT_SLOTS {architecture['output_slots']}
#define MOVE_STRIDE {architecture['move_stride']}
#define QUANT_SHIFT {architecture['quant_shift']}
#define MAX_FEATURES {architecture['max_features_per_record']}
"""


def _run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess command and return the CompletedProcess."""
    return subprocess.run(cmd, **kwargs)


def _run_train_loop(
    tmp_path: Path,
    weights_path: Path,
    seed: int,
    bench_seconds: float,
    random_seconds: float,
    selfplay_games: int,
    epochs: int,
    batch_size: int,
    max_train: int,
    max_val: int,
    timeout: float,
) -> dict:
    """Run train_loop.py once and return its parsed JSON result dict."""
    train_cmd = [
        sys.executable,
        str(tmp_path / "train_loop.py"),
        str(tmp_path / "seed_engine.cpp"),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--max-train-samples", str(max_train),
        "--max-val-samples", str(max_val),
        "--selfplay-games", str(selfplay_games),
        "--bench-seconds", str(bench_seconds),
        "--random-seconds", str(random_seconds),
        "--weights-path", str(weights_path),
        "--seed", str(seed),
        "--output", "json",
    ]
    proc = _run(
        train_cmd,
        cwd=str(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    train_output = (proc.stdout or b"").decode(errors="replace")
    train_err = (proc.stderr or b"").decode(errors="replace")

    result: dict = {"_stdout": train_output, "_stderr": train_err}
    if proc.returncode != 0:
        result["error"] = f"micro-training failed (exit {proc.returncode}): {train_err[:500]}"
        return result

    try:
        parsed = json.loads(train_output.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        result["error"] = f"could not parse train_loop JSON output: {train_output[:500]}"
        return result

    if not isinstance(parsed, dict):
        result["error"] = "train_loop JSON output is not a dict"
        return result

    if parsed.get("error"):
        result["error"] = f"micro-training reported error: {parsed['error']}"
        return result

    return parsed




def _aggregate_results(tier1_results: list) -> dict:
    """Aggregate multi-seed Tier 1 metrics into mean/std and final metric values.

    Args:
        tier1_results: List of result dicts returned by _run_train_loop.

    Returns:
        Dict with mean_*, std_*, and raw keys for win_rate, nodes_per_second,
        draw_rate, avg_game_length, and val_loss.
    """
    import numpy as np

    metrics: dict = {}
    for key in ("win_rate", "nodes_per_second", "draw_rate", "avg_game_length", "val_loss"):
        vals = [float(r.get(key, 0.0)) for r in tier1_results]
        metrics[f"mean_{key}"] = float(np.mean(vals))
        metrics[f"std_{key}"] = float(np.std(vals))
        metrics[key] = metrics[f"mean_{key}"]
    return metrics


def _tier2_gate_passes(win_rate: float, run_tier2: bool, tier2_threshold: float) -> bool:
    """Return True if Tier-2 long-horizon validation should run."""
    return run_tier2 and win_rate >= tier2_threshold


def _apply_tier2_metrics(metrics: dict, t2_res: dict, tier2_replaces: bool) -> dict:
    """Merge Tier-2 results into metrics, optionally replacing Tier-1 values."""
    for key in ("win_rate", "nodes_per_second", "draw_rate", "avg_game_length", "val_loss"):
        metrics[f"tier2_{key}"] = float(t2_res.get(key, 0.0))
    if tier2_replaces:
        for key in ("win_rate", "nodes_per_second", "draw_rate", "avg_game_length", "val_loss"):
            metrics[key] = metrics[f"tier2_{key}"]
    return metrics


def evaluate(program_path: str) -> dict:
    """OpenEvolve evaluator entry point.

    Args:
        program_path: Path to the mutated aetherstate_bundle.py file.

    Returns:
        EvaluationResult dictionary with trainability metrics.
    """
    start_time = time.time()
    metrics: dict = {
        "fitness": 0.0,
        "combined_score": 0.0,
        "compile_ok": 0.0,
        "generate_data_ok": 0.0,
        "train_start_ok": 0.0,
        "weights_export_ok": 0.0,
        "weights_load_ok": 0.0,
        "nodes_per_second": 0.0,
        "trainable": 0.0,
        "error": "",
    }
    artifacts: dict = {}

    # -----------------------------------------------------------------------
    # 1. Parse bundle
    # -----------------------------------------------------------------------
    try:
        bundle = _parse_bundle(program_path)
    except ValueError as e:
        metrics["error"] = f"bundle parse error: {e}"
        return EvaluationResult(metrics=metrics, artifacts=artifacts).to_dict()

    architecture = bundle["architecture"]

    # Persist the research notebook so OpenEvolve can store it with the program.
    if bundle.get("research_notebook"):
        artifacts["research_notebook"] = bundle["research_notebook"]

    # -----------------------------------------------------------------------
    # 2. Validate architecture
    # -----------------------------------------------------------------------
    arch_ok, arch_err = _validate_architecture(architecture)
    if not arch_ok:
        metrics["error"] = f"architecture validation failed: {arch_err}"
        return EvaluationResult(metrics=metrics, artifacts=artifacts).to_dict()

    # -----------------------------------------------------------------------
    # 3. Stage files in a temp directory
    # -----------------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="aetherstate_eval_") as tmpdir:
        tmp_path = Path(tmpdir)

        # Write arch_constants.py for the Python training loop
        arch_constants_py = _build_arch_constants_py(architecture)
        (tmp_path / "arch_constants.py").write_text(arch_constants_py, encoding="utf-8")

        # Write seed_engine.cpp with architecture header prepended
        cpp_header = _build_cpp_architecture_header(architecture)
        seed_engine_cpp = bundle["seed_engine_cpp"]
        (tmp_path / "seed_engine.cpp").write_text(
            cpp_header + seed_engine_cpp, encoding="utf-8"
        )

        # Write train_loop.py
        train_loop_py = bundle["train_loop_py"]
        (tmp_path / "train_loop.py").write_text(train_loop_py, encoding="utf-8")

        # Copy required headers
        missing_headers = []
        for hdr in NEEDED_HEADERS:
            src = TASKS_DIR / hdr
            if not src.is_file():
                missing_headers.append(str(src))
                continue
            shutil.copy2(src, tmp_path / hdr)
        if missing_headers:
            metrics["error"] = f"required headers missing: {missing_headers}"
            return EvaluationResult(metrics=metrics, artifacts=artifacts).to_dict()

        binary_path = tmp_path / "aetherstate"
        weights_path = tmp_path / "weights.bin"

        # -------------------------------------------------------------------
        # 4. Compile C++ engine
        # -------------------------------------------------------------------
        compile_timeout = 45.0
        compile_start = time.time()
        compile_ok = False
        for flags in ("-mavx2 -mavx512f", ""):
            cmd = (
                ["g++", "-std=c++17", "-O3"]
                + (flags.split() if flags else [])
                + [str(tmp_path / "seed_engine.cpp"), "-o", str(binary_path)]
            )
            try:
                proc = _run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=compile_timeout,
                )
            except subprocess.TimeoutExpired:
                metrics["error"] = "C++ compile timed out"
                return EvaluationResult(metrics=metrics, artifacts=artifacts).to_dict()
            if proc.returncode == 0:
                compile_ok = True
                break
        if not compile_ok:
            artifacts["stderr"] = (proc.stderr or b"").decode(errors="replace")
            metrics["error"] = f"C++ compile failed: {artifacts['stderr'][:500]}"
            return EvaluationResult(metrics=metrics, artifacts=artifacts).to_dict()

        metrics["compile_ok"] = 1.0
        compile_elapsed = time.time() - compile_start

        # -------------------------------------------------------------------
        # 5. Python syntax check
        # -------------------------------------------------------------------
        py_syntax = _run(
            [sys.executable, "-m", "py_compile", str(tmp_path / "train_loop.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10.0,
        )
        if py_syntax.returncode != 0:
            artifacts["py_syntax_error"] = (py_syntax.stderr or b"").decode(errors="replace")
            metrics["error"] = f"Python syntax error: {artifacts['py_syntax_error'][:500]}"
            return EvaluationResult(metrics=metrics, artifacts=artifacts).to_dict()

        # -------------------------------------------------------------------
        # 6. Generate a small amount of data from the C++ engine (trainability)
        # -------------------------------------------------------------------
        data_gen = _run(
            [str(binary_path), "generate_data", "10"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30.0,
        )
        if data_gen.returncode != 0:
            artifacts["generate_data_stderr"] = (data_gen.stderr or b"").decode(errors="replace")
            metrics["error"] = f"generate_data failed: {artifacts['generate_data_stderr'][:500]}"
            return EvaluationResult(metrics=metrics, artifacts=artifacts).to_dict()

        if not data_gen.stdout:
            metrics["error"] = "generate_data produced no output"
            return EvaluationResult(metrics=metrics, artifacts=artifacts).to_dict()

        # Verify the output is a non-empty binary stream of training records.
        record_size = (
            4 + 4 + 4 * architecture["max_features_per_record"] + 4 + 4 + 4 + 4
        )
        if len(data_gen.stdout) < record_size:
            metrics["error"] = "generate_data produced fewer bytes than a single record"
            return EvaluationResult(metrics=metrics, artifacts=artifacts).to_dict()

        metrics["generate_data_ok"] = 1.0        # Tunables for the micro-training gate (env-overridable for slow/fast hosts).
        bench_seconds = float(os.environ.get("AETHERSTATE_BENCH_SECONDS", "2"))
        random_seconds = float(os.environ.get("AETHERSTATE_RANDOM_SECONDS", "5"))
        selfplay_games = int(os.environ.get("AETHERSTATE_SELFPLAY_GAMES", "64"))

        # -------------------------------------------------------------------
        # 7. Micro-train: compile + generate + train + export weights + benchmark
        # -------------------------------------------------------------------
        # Multi-seed and two-tier configuration.  These are env-overridable so
        # the user can trade evaluation cost for measurement stability without
        # editing code.
        num_seeds = int(os.environ.get("AETHERSTATE_NUM_SEEDS", "1"))
        per_seed_timeout = float(os.environ.get("AETHERSTATE_PER_SEED_TIMEOUT", "180.0"))
        tier2_threshold = float(os.environ.get("AETHERSTATE_TIER2_WIN_RATE_THRESHOLD", "0.6"))
        run_tier2 = os.environ.get("AETHERSTATE_RUN_TIER2", "1") == "1"
        tier2_replaces = os.environ.get("AETHERSTATE_TIER2_REPLACES_FITNESS", "0") == "1"

        train_start = time.time()
        tier1_results = []
        for seed in range(max(1, num_seeds)):
            # Each seed gets its own fixed timeout budget so multi-seed runs do
            # not progressively starve later seeds.  Use a seed-specific weights
            # file so later seeds cannot leave stale state from earlier seeds.
            seed_weights_path = weights_path.with_suffix(f".seed_{seed}.bin")
            seed_timeout = min(per_seed_timeout, max(10.0, 240.0 - (time.time() - train_start)))
            res = _run_train_loop(
                tmp_path,
                seed_weights_path,
                seed=seed,
                bench_seconds=bench_seconds,
                random_seconds=random_seconds,
                selfplay_games=selfplay_games,
                epochs=1,
                batch_size=256,
                max_train=2048,
                max_val=512,
                timeout=seed_timeout,
            )
            if res.get("error"):
                metrics["error"] = f"micro-training seed {seed} failed: {res['error']}"
                artifacts["train_output"] = res.get("_stdout", "")
                artifacts["train_stderr"] = res.get("_stderr", "")
                return EvaluationResult(metrics=metrics, artifacts=artifacts).to_dict()
            tier1_results.append(res)

        # Use the first seed's artifacts for debugging.
        artifacts["train_output"] = tier1_results[0].get("_stdout", "")
        artifacts["train_stderr"] = tier1_results[0].get("_stderr", "")
        metrics["train_start_ok"] = 1.0

        # -------------------------------------------------------------------
        # 8. Verify weights were exported and load in C++
        # -------------------------------------------------------------------
        # Validate the last seed's weights (highest seed), which gives the
        # cleanest view of the current implementation's export/load behavior.
        weights_path = weights_path.with_suffix(f".seed_{max(0, max(1, num_seeds) - 1)}.bin")
        if not weights_path.exists():
            metrics["error"] = "weights.bin was not exported"
            return EvaluationResult(metrics=metrics, artifacts=artifacts).to_dict()

        if weights_path.stat().st_size == 0:
            metrics["error"] = "weights.bin is empty"
            return EvaluationResult(metrics=metrics, artifacts=artifacts).to_dict()

        # We deliberately do NOT enforce a fixed weight-file size.  The LLM is
        # allowed to change the network topology (add/remove layers, change
        # shapes) as long as the C++ engine can load the exported weights.
        # The load_weights check below is the real guardrail.
        metrics["weights_export_ok"] = 1.0

        load_check = _run(
            [str(binary_path), "load_weights", str(weights_path), "bench_time", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30.0,
        )
        if load_check.returncode != 0:
            artifacts["load_weights_stderr"] = (load_check.stderr or b"").decode(errors="replace")
            metrics["error"] = f"load_weights failed: {artifacts['load_weights_stderr'][:500]}"
            return EvaluationResult(metrics=metrics, artifacts=artifacts).to_dict()

        metrics["weights_load_ok"] = 1.0

        # -------------------------------------------------------------------
        # 9. Aggregate multi-seed Tier 1 metrics
        # -------------------------------------------------------------------
        metrics.update(_aggregate_results(tier1_results))

        # -------------------------------------------------------------------
        # 10. Tier 2 long-horizon validation for promising candidates
        # -------------------------------------------------------------------
        if _tier2_gate_passes(metrics["win_rate"], run_tier2, tier2_threshold):
            t2_bench = float(os.environ.get("AETHERSTATE_TIER2_BENCH_SECONDS", "10"))
            t2_random = float(os.environ.get("AETHERSTATE_TIER2_RANDOM_SECONDS", "30"))
            t2_selfplay = int(os.environ.get("AETHERSTATE_TIER2_SELFPLAY_GAMES", "256"))
            t2_timeout = float(os.environ.get("AETHERSTATE_TIER2_TIMEOUT", "600"))
            remaining = max(10.0, t2_timeout - (time.time() - train_start))
            # Use a fresh seed for Tier 2 so it does not merely reproduce a
            # previously seen random configuration.  A fixed large base keeps it
            # distinct from any Tier-1 seed count.
            t2_seed = 1000 + max(1, num_seeds)
            t2_res = _run_train_loop(
                tmp_path,
                weights_path.with_suffix(".tier2.bin"),
                seed=t2_seed,
                bench_seconds=t2_bench,
                random_seconds=t2_random,
                selfplay_games=t2_selfplay,
                epochs=1,
                batch_size=256,
                max_train=8192,
                max_val=2048,
                timeout=remaining,
            )
            if t2_res.get("error"):
                # Tier 2 failure is not fatal; just record it.
                metrics["tier2_error"] = str(t2_res["error"])
            else:
                metrics = _apply_tier2_metrics(metrics, t2_res, tier2_replaces)

        # -------------------------------------------------------------------
        # 11. Assemble final metrics
        # -------------------------------------------------------------------
        trainable = all(
            metrics[k] == 1.0
            for k in (
                "compile_ok",
                "generate_data_ok",
                "train_start_ok",
                "weights_export_ok",
                "weights_load_ok",
            )
        )
        metrics["trainable"] = 1.0 if trainable else 0.0
        metrics["compile_time"] = compile_elapsed
        metrics["train_time"] = time.time() - train_start

        # Architecture signature: a coarse hash used as a MAP-Elites axis to
        # preserve diversity of architectures.  Different topologies land in
        # different bins, so a novel architecture is not immediately evicted by
        # a slightly higher-fitness conventional one.
        arch_signature = _architecture_signature(architecture)
        metrics["architecture_signature"] = arch_signature

        # Fitness combines the (possibly tier-2-replaced) win rate and speed.
        speed_score = metrics["nodes_per_second"] / 1_000_000.0
        fitness = 10.0 * metrics["win_rate"] + speed_score
        if not trainable:
            fitness = 0.0
        metrics["fitness"] = fitness
        metrics["combined_score"] = fitness

        return EvaluationResult(metrics=metrics, artifacts=artifacts).to_dict()


def _architecture_signature(architecture: dict) -> int:
    """Return a coarse, stable hash of the architecture dict.

    This is used as a MAP-Elites feature dimension to preserve diverse
    architectures.  It intentionally ignores the mutable code strings and
    focuses on the architecture dimensions and version.
    """
    import hashlib
    keys = sorted(k for k in architecture.keys() if k not in {"weight_magic"})
    payload = ",".join(f"{k}={architecture[k]}" for k in keys)
    return int(hashlib.md5(payload.encode("utf-8")).hexdigest(), 16) % 1000


def _parse_key_value_output(text: str) -> dict:
    """Parse key=value lines printed by the C++ engine."""
    result: dict = {}
    for line in text.strip().splitlines():
        for part in line.strip().split():
            if "=" in part:
                key, val = part.split("=", 1)
                key = key.strip().lower()
                val = val.strip()
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
    return result


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "aetherstate_bundle.py"
    print(json.dumps(evaluate(path), indent=2))
