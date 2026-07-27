#!/usr/bin/env python3
"""
AetherState Train Loop
----------------------
PyTorch-based zero-knowledge self-play reinforcement learning pipeline for the
AetherState pure-AI chess engine.

Usage:
    python train_loop.py <path_to_seed_engine.cpp> [options]

Options:
    --compile-timeout <seconds>   Max time allowed for g++ compilation (default 15)
    --bench-seconds <N>           Duration of the speed micro-benchmark (default 90)
    --random-seconds <N>          Duration of convergence benchmark vs random (default 180)
    --selfplay-games <N>          Self-play games for data generation (default 256)
    --epochs <N>                  Training epochs over generated data (default 5)
    --batch-size <N>              Training batch size (default 64)
    --lr <float>                  Learning rate (default 1e-3)
    --output <json|text>          Output format (default text)
    --no-train                    Skip training; only compile and benchmark the seed.

The script:
  1. Compiles the supplied C++ engine.
  2. Generates self-play training data using the engine.
  3. Trains a PyTorch policy+value network with REINFORCE.
  4. Quantizes and exports int8 weights to a binary file the C++ engine loads.
  5. Loads the trained weights into the C++ engine.
  6. Runs speed and win-rate benchmarks and reports the MAP-Elites metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# PyTorch is required for the RL training loop.  If it is missing, the script
# still compiles and benchmarks the raw C++ seed so the evaluator can score it.
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Configuration that must stay in sync with seed_engine.cpp
# ---------------------------------------------------------------------------
INPUT_FEATURES = 768
ACCUMULATOR_SIZE = 256
HIDDEN_SIZE = 32
OUTPUT_SLOTS = 4096
WEIGHT_MAGIC = b"AESTATEW"

# C++ struct NeuralNet size including 64-byte alignment padding.
# Computed as the next multiple of 64 above the raw array sum.
_RAW_NET_SIZE = (
    INPUT_FEATURES * ACCUMULATOR_SIZE * 1  # w1
    + ACCUMULATOR_SIZE * 2                 # b1
    + ACCUMULATOR_SIZE * HIDDEN_SIZE * 1   # w2
    + HIDDEN_SIZE * 2                      # b2
    + HIDDEN_SIZE * OUTPUT_SLOTS * 1         # w3
    + OUTPUT_SLOTS * 2                     # b3
    + HIDDEN_SIZE * 1                        # wv
    + 2                                      # bv
)
NET_SIZE = ((_RAW_NET_SIZE + 63) // 64) * 64

# TrainingRecord binary layout used by the C++ generate_data mode.
TRAINING_RECORD_DTYPE = np.dtype([
    ("side_to_move", "<i4"),
    ("n_features", "<i4"),
    ("features", "<i4", 32),
    ("from_", "<i4"),
    ("to_", "<i4"),
    ("promo", "<i4"),
    ("outcome", "<i4"),
])


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------
def compile_engine(source_path: Path, binary_path: Path, timeout: float = 15.0) -> dict:
    """Compile the C++ source. Return status and elapsed time."""
    start = time.time()
    # Try the requested SIMD flags first; if the host compiler rejects them,
    # fall back to a plain C++17 build.  The evaluator treats any compilation
    # longer than `timeout` as a failure.
    for flags in ("-mavx2 -mavx512f", ""):
        cmd = (
            ["g++", "-std=c++17", "-O3"]
            + (flags.split() if flags else [])
            + [str(source_path), "-o", str(binary_path)]
        )
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "elapsed": time.time() - start, "error": "compile timeout"}
        if proc.returncode == 0:
            return {"ok": True, "elapsed": time.time() - start, "flags": flags or "none"}
    return {
        "ok": False,
        "elapsed": time.time() - start,
        "error": (proc.stderr or b"unknown compile error").decode(errors="replace"),
    }


def run_binary(binary_path: Path, *args: str) -> dict:
    """Run the engine with the given mode/args and parse key=value output."""
    cmd = [str(binary_path), *args]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr or "engine crashed"}
    result: dict = {}
    for line in proc.stdout.strip().splitlines():
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


# ---------------------------------------------------------------------------
# PyTorch model matching the C++ DFCAM architecture
# ---------------------------------------------------------------------------
if HAS_TORCH:
    class AetherStateNet(nn.Module):  # type: ignore
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(INPUT_FEATURES, ACCUMULATOR_SIZE, bias=True)
            self.fc2 = nn.Linear(ACCUMULATOR_SIZE, HIDDEN_SIZE, bias=True)
            self.policy_head = nn.Linear(HIDDEN_SIZE, OUTPUT_SLOTS, bias=True)
            self.value_head = nn.Linear(HIDDEN_SIZE, 1, bias=True)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Accumulator layer (no activation, matching the C++ accumulator).
            acc = self.fc1(x)
            hidden = F.relu(self.fc2(acc))
            policy = self.policy_head(hidden)
            value = self.value_head(hidden).squeeze(-1)
            return policy, value


# ---------------------------------------------------------------------------
# Data generation and ingestion
# ---------------------------------------------------------------------------
def generate_data(binary_path: Path, games: int, work_dir: Path) -> Path:
    """Run the C++ engine in generate_data mode and write records to disk."""
    data_path = work_dir / "training_data.bin"
    proc = subprocess.run(
        [str(binary_path), "generate_data", str(games)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"generate_data failed: {proc.stderr.decode(errors='replace')}")
    data_path.write_bytes(proc.stdout)
    return data_path


def load_training_data(data_path: Path) -> np.ndarray:
    """Load binary training records produced by the C++ engine."""
    data = np.fromfile(data_path, dtype=TRAINING_RECORD_DTYPE)
    return data


def records_to_tensors(records: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert numpy records to PyTorch tensors.

    Returns:
        features: LongTensor of active feature indices (padded with -1).
        actions: LongTensor of chosen action indices (from * 64 + to).
        outcomes: FloatTensor of final game outcomes from side_to_move perspective.
        masks: BoolTensor indicating valid feature slots.
    """
    n = len(records)
    features = torch.from_numpy(records["features"].astype(np.int64)).to(device)
    n_features = torch.from_numpy(records["n_features"].astype(np.int64)).to(device)
    actions = torch.from_numpy((records["from_"] * 64 + records["to_"]).astype(np.int64)).to(device)
    outcomes = torch.from_numpy(records["outcome"].astype(np.float32)).to(device)

    # Build a mask so padding with -1 is ignored when we scatter.
    mask = torch.arange(32, device=device).unsqueeze(0) < n_features.unsqueeze(1)
    return features, actions, outcomes, mask


# ---------------------------------------------------------------------------
# Weight export (int8 / int16) for the C++ engine
# ---------------------------------------------------------------------------
def _pack_int8(arr: np.ndarray) -> bytes:
    return np.clip(np.rint(arr), -128, 127).astype(np.int8).tobytes()


def _pack_int16(arr: np.ndarray) -> bytes:
    return np.clip(np.rint(arr), -32768, 32767).astype(np.int16).tobytes()


def export_weights(model: nn.Module, path: Path) -> None:
    """Export trained PyTorch weights in the exact C++ NeuralNet layout."""
    state = model.state_dict()
    # fc1.weight is (256, 768); C++ w1 is (768, 256). Transpose to match.
    w1 = state["fc1.weight"].detach().cpu().numpy().T
    b1 = state["fc1.bias"].detach().cpu().numpy()
    w2 = state["fc2.weight"].detach().cpu().numpy().T
    b2 = state["fc2.bias"].detach().cpu().numpy()
    w3 = state["policy_head.weight"].detach().cpu().numpy().T
    b3 = state["policy_head.bias"].detach().cpu().numpy()
    wv = state["value_head.weight"].detach().cpu().numpy().squeeze()
    bv = state["value_head.bias"].detach().cpu().numpy().item()

    payload = b""
    payload += _pack_int8(w1)
    payload += _pack_int16(b1)
    payload += _pack_int8(w2)
    payload += _pack_int16(b2)
    payload += _pack_int8(w3)
    payload += _pack_int16(b3)
    payload += _pack_int8(wv)
    payload += _pack_int16(np.array([bv], dtype=np.float32))

    # Pad to the C++ struct size (multiple of 64).
    if len(payload) < NET_SIZE:
        payload += b"\x00" * (NET_SIZE - len(payload))
    elif len(payload) > NET_SIZE:
        raise RuntimeError(f"Exported weights too large: {len(payload)} > {NET_SIZE}")

    with open(path, "wb") as f:
        f.write(WEIGHT_MAGIC)
        f.write(payload)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    features: torch.Tensor,
    actions: torch.Tensor,
    outcomes: torch.Tensor,
    mask: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    """Train the policy+value network with REINFORCE on generated data."""
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n = features.size(0)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            batch_features = features[idx]
            batch_mask = mask[idx]
            batch_actions = actions[idx]
            batch_outcomes = outcomes[idx]

            # Build sparse multi-hot input of shape (B, 768).
            bsz = batch_features.size(0)
            x = torch.zeros(bsz, INPUT_FEATURES, device=device)
            if bsz > 0:
                valid = batch_features >= 0
                if valid.any():
                    rows = torch.arange(bsz, device=device).unsqueeze(1).expand(-1, 32)[valid]
                    cols = batch_features[valid]
                    x[rows, cols] = 1.0

            policy_logits, value = model(x)

            # REINFORCE policy loss: maximize log-prob of chosen action weighted
            # by the final outcome from the acting side's perspective.
            log_probs = F.log_softmax(policy_logits, dim=-1)
            chosen_log_probs = log_probs[torch.arange(bsz, device=device), batch_actions]
            policy_loss = -(chosen_log_probs * batch_outcomes).mean()

            # Value loss: regress the value head on the same outcome.
            value_loss = F.mse_loss(value, batch_outcomes)

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            batches += 1

        if epoch == epochs - 1:
            print(
                f"Epoch {epoch+1}/{epochs}  "
                f"policy_loss={total_policy_loss/max(batches,1):.4f}  "
                f"value_loss={total_value_loss/max(batches,1):.4f}"
            )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AetherState PyTorch RL train loop")
    parser.add_argument("source_path", type=Path, help="Path to seed_engine.cpp")
    parser.add_argument("--compile-timeout", type=float, default=60.0)
    parser.add_argument("--bench-seconds", type=float, default=90.0)
    parser.add_argument("--random-seconds", type=float, default=180.0)
    parser.add_argument("--selfplay-games", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", choices=["json", "text"], default="text")
    parser.add_argument("--no-train", action="store_true", help="Skip training; only benchmark the seed.")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip the final C++ speed and random-opponent benchmarks. Useful when train_loop.py is used as a quick trainability/structural gate.")
    args = parser.parse_args(argv)

    source_path = args.source_path.resolve()
    if not source_path.exists():
        print(json.dumps({"error": f"source not found: {source_path}", "fitness": 0.0}))
        return 1

    work_dir = Path(tempfile.mkdtemp(prefix="aetherstate_"))
    binary_path = work_dir / "aetherstate"
    weights_path = work_dir / "weights.bin"
    result: dict = {"source": str(source_path)}

    # 1) Compile
    compile_info = compile_engine(source_path, binary_path, timeout=args.compile_timeout)
    result["compile_ok"] = compile_info["ok"]
    result["compile_time"] = compile_info.get("elapsed", 0.0)
    result["compile_flags"] = compile_info.get("flags", "")
    if not compile_info["ok"]:
        result["fitness"] = 0.0
        result["compile_error"] = compile_info.get("error", "")
        print(_format(result, args.output))
        return 0

    # If no training is requested, just benchmark the raw seed.
    if args.no_train or not HAS_TORCH:
        if not HAS_TORCH:
            print("Warning: PyTorch not available; running in benchmark-only mode.", file=sys.stderr)
        speed = run_binary(binary_path, "bench_time", str(args.bench_seconds))
        conv = run_binary(binary_path, "random_time", str(args.random_seconds))
        result.update(_metrics(speed, conv))
        print(_format(result, args.output))
        return 0

    # 2) Generate self-play data.
    data_path = generate_data(binary_path, args.selfplay_games, work_dir)

    # 3) Load and prepare data.
    records = load_training_data(data_path)
    if len(records) == 0:
        print("Warning: no training records generated.", file=sys.stderr)
        result.update({"nodes_per_second": 0.0, "win_rate": 0.0, "fitness": 0.0})
        print(_format(result, args.output))
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, actions, outcomes, mask = records_to_tensors(records, device)

    # 4) Train.
    model = AetherStateNet().to(device)
    train_model(
        model,
        features,
        actions,
        outcomes,
        mask,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # 5) Export int8 weights.
    export_weights(model, weights_path)

    # 6) Verify the C++ engine can load the exported weights.  When
    #    --no-benchmark is set we only do a tiny bench_time 0 dry-run to keep
    #    the gate fast; otherwise run the full requested benchmarks.
    if args.no_benchmark:
        load_check = run_binary(
            binary_path, "load_weights", str(weights_path), "bench_time", "0"
        )
        if "error" in load_check:
            result["error"] = f"load_weights dry-run failed: {load_check['error']}"
            print(_format(result, args.output))
            return 1
        result["no_benchmark"] = True
    else:
        speed = run_binary(binary_path, "load_weights", str(weights_path), "bench_time", str(args.bench_seconds))
        conv = run_binary(binary_path, "load_weights", str(weights_path), "random_time", str(args.random_seconds))
        result.update(_metrics(speed, conv))
    result["trained_weights"] = str(weights_path)
    print(_format(result, args.output))
    return 0


def _metrics(speed: dict, conv: dict) -> dict:
    nodes_per_second = speed.get("nodes_per_second", 0.0)
    win_rate = conv.get("win_rate", 0.0)
    # Fitness used for selection; MAP-Elites axes are reported separately.
    fitness = (win_rate * 1000.0) + (nodes_per_second / 100000.0)
    out = {
        "nodes_per_second": nodes_per_second,
        "win_rate": win_rate,
        "fitness": fitness,
    }
    if "error" in speed:
        out["speed_error"] = speed["error"]
    if "error" in conv:
        out["conv_error"] = conv["error"]
    return out


def _format(result: dict, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(result)
    lines = ["AetherState Train Loop Results", "=" * 40]
    for key, value in result.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
