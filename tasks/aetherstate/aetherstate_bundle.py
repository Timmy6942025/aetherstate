"""
AetherState OpenEvolve Bundle
------------------------------
This file is the single artifact that OpenEvolve mutates.  It contains:
  - ARCHITECTURE: a single source of truth for all neural-network dimensions.
  - SEED_ENGINE_CPP: the mutable C++ inference engine source.
  - TRAIN_LOOP_PY: the mutable PyTorch training pipeline source.

The AetherState evaluator (evaluate.py) parses this bundle, injects the
architecture constants into both language-specific files, and runs a fast
trainability gate.  Because the constants are injected from ARCHITECTURE,
the C++ engine and Python training loop can never drift out of sync.

When editing this file by hand, keep the raw triple-quoted strings intact.
When OpenEvolve mutates it, it may change the ARCHITECTURE dict and any code
inside the two string blocks using SEARCH/REPLACE diffs.
"""

# ---------------------------------------------------------------------------
# Single source of truth for the neural-network architecture.
# The evaluator injects these into seed_engine.cpp (as C++ macros) and into
# train_loop.py (as Python constants via arch_constants.py).
# ---------------------------------------------------------------------------
ARCHITECTURE = {
    "input_features": 768,    # 12 bitboards * 64 squares; must match C++ code
    "accumulator_size": 256,  # Layer 1 output / accumulator width
    "hidden_size": 32,        # Layer 2 hidden units
    "output_slots": 4096,     # 64 from-squares * 64 to-squares; fixed by encoding
    "move_stride": 64,        # Must match output_slots encoding
    "quant_shift": 7,         # Fixed-point right shift for int8 activations
    "weight_magic": "AESTATEW",  # 8-byte weight-file header; DO NOT CHANGE
    "max_features_per_record": 32,  # TrainingRecord.features[] size in C++
    "weight_version": 1,      # Increment when you change the C++ NeuralNet struct topology
}


# ---------------------------------------------------------------------------
# Living research notebook.
# The LLM records hypotheses, expected effects, and falsification criteria
# here before each mutation.  It is passed back in subsequent prompts so the
# loop builds institutional memory.
# ---------------------------------------------------------------------------
RESEARCH_NOTEBOOK = r'''
## Initial seed hypothesis
- Hypothesis: The DFCAM accumulator architecture with int8 weights and a
  small hidden layer can learn enough chess to beat a random opponent after a
  short self-play run.
- Expected effect: Baseline win_rate against random >> 0.5 and inference
  speed > 100k nodes/sec.
- Falsification: If win_rate <= 0.5 or training crashes, the seed is not
  trainable and the architecture must change.
'''


# ---------------------------------------------------------------------------
# Mutable C++ inference engine.
# Architecture constants are injected as #define macros before compilation.
# ---------------------------------------------------------------------------
SEED_ENGINE_CPP = r'''/*
 * AetherState Seed Engine - DFCAM Inference Core
 *
 * This file is the OpenEvolve seed.  It contains ONLY the mutable neural
 * network, accumulator, and policy code.  All fixed chess rules, move
 * generation, and CLI harness logic live in chess_runtime.hpp.
 *
 * The block between
 *     // # EVOLVE-BLOCK-START
 *     // # EVOLVE-BLOCK-END
 * is the only region OpenEvolve should modify.
 *
 * Architecture (DFCAM - Deep Fully-Connected Accumulator Matrix):
 *   Input:   768 sparse bitboard features (12 bitboards * 64 squares)
 *   Layer 1: Accumulator(256)  -- persistent, updated by differential sparse add/remove
 *   Layer 2: Hidden(32)        -- ReLU
 *   Output:  Move logits(4096) -- masked by legal moves, argmax selected
 *   Value:   Single scalar from same trunk
 *
 * Quantization: int8 weights, int32 activations, fixed-point shift=7.
 * SIMD: uses the platform-agnostic SIMD_DOT16() macro from chess_runtime.hpp,
 * compiling to ARM NEON on this host and AVX2/AVX-512 on x86_64 cloud nodes.
 *
 * Architecture constants (INPUT_FEATURES, ACCUMULATOR_SIZE, HIDDEN_SIZE,
 * OUTPUT_SLOTS, MOVE_STRIDE, QUANT_SHIFT) are injected by the evaluator as
 * preprocessor macros before this file is compiled. Do NOT define them here.
 */

#include "chess_runtime.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <vector>

// # EVOLVE-BLOCK-START

// ---------------------------------------------------------------------------
// Network hyper-parameters and quantization constants
// ---------------------------------------------------------------------------
// The following are injected as #define macros by the AetherState evaluator:
//   INPUT_FEATURES   = 768  (12 bitboards * 64 squares)
//   ACCUMULATOR_SIZE = 256
//   HIDDEN_SIZE      = 32
//   OUTPUT_SLOTS     = 4096 (from-square 64 * to-square 64)
//   MOVE_STRIDE      = 64
//   QUANT_SHIFT      = 7
//
// Do not define them in this file; the evaluator prepends them before compile.
// You may change how they are used (e.g., SIMD loops, layer shapes), but the
// evaluator will reject combinations that break the weight-file format or
// exceed safe size limits.

// ---------------------------------------------------------------------------
// Neural network weight storage (int8, row-major)
// ---------------------------------------------------------------------------
struct alignas(64) NeuralNet {
    // Layer 1: input -> accumulator
    int8_t  w1[INPUT_FEATURES][ACCUMULATOR_SIZE];
    int16_t b1[ACCUMULATOR_SIZE];
    // Layer 2: accumulator -> hidden
    int8_t  w2[ACCUMULATOR_SIZE][HIDDEN_SIZE];
    int16_t b2[HIDDEN_SIZE];
    // Layer 3: hidden -> output move logits
    int8_t  w3[HIDDEN_SIZE][OUTPUT_SLOTS];
    int16_t b3[OUTPUT_SLOTS];
    // Value head: hidden -> value scalar
    int8_t  wv[HIDDEN_SIZE];
    int16_t bv;
};

NeuralNet g_net;
std::mt19937_64 g_rng(12345);
std::uniform_int_distribution<int> g_weight_dist(-128, 127);

static inline int clamp16(int32_t x) {
    if (x < -32768) return -32768;
    if (x > 32767) return 32767;
    return x;
}

static inline int16_t relu16(int16_t x) { return x > 0 ? x : 0; }

// Initialize (or re-initialize) the network with small random int8 weights.
void init_net() {
    for (int i = 0; i < INPUT_FEATURES; ++i)
        for (int j = 0; j < ACCUMULATOR_SIZE; ++j)
            g_net.w1[i][j] = (int8_t)g_weight_dist(g_rng);
    for (int j = 0; j < ACCUMULATOR_SIZE; ++j)
        g_net.b1[j] = (int16_t)(g_weight_dist(g_rng) % 256 - 128);

    for (int j = 0; j < ACCUMULATOR_SIZE; ++j)
        for (int k = 0; k < HIDDEN_SIZE; ++k)
            g_net.w2[j][k] = (int8_t)g_weight_dist(g_rng);
    for (int k = 0; k < HIDDEN_SIZE; ++k)
        g_net.b2[k] = (int16_t)(g_weight_dist(g_rng) % 256 - 128);

    for (int k = 0; k < HIDDEN_SIZE; ++k)
        for (int l = 0; l < OUTPUT_SLOTS; ++l)
            g_net.w3[k][l] = (int8_t)g_weight_dist(g_rng);
    for (int l = 0; l < OUTPUT_SLOTS; ++l)
        g_net.b3[l] = (int16_t)(g_weight_dist(g_rng) % 256 - 128);

    for (int k = 0; k < HIDDEN_SIZE; ++k)
        g_net.wv[k] = (int8_t)g_weight_dist(g_rng);
    g_net.bv = (int16_t)(g_weight_dist(g_rng) % 256 - 128);
}

// ---------------------------------------------------------------------------
// Sparse accumulator
// ---------------------------------------------------------------------------
// The accumulator is Layer 1.  It can be built from scratch for a position,
// or updated differentially when a move is made.  The differential update is:
//   Accumulator(t) = Accumulator(t-1) + W[:, added] - W[:, removed]
// ---------------------------------------------------------------------------
struct Accumulator {
    int16_t acc[ACCUMULATOR_SIZE];
    bool valid;
    std::vector<int> active_features;

    void reset() {
        std::memcpy(acc, g_net.b1, sizeof(g_net.b1));
        valid = true;
    }

    void add_feature(int idx) {
        if (idx < 0 || idx >= INPUT_FEATURES) return;
        for (int j = 0; j < ACCUMULATOR_SIZE; ++j) {
            acc[j] += g_net.w1[idx][j];
        }
        active_features.push_back(idx);
    }

    void remove_feature(int idx) {
        if (idx < 0 || idx >= INPUT_FEATURES) return;
        for (int j = 0; j < ACCUMULATOR_SIZE; ++j) {
            acc[j] -= g_net.w1[idx][j];
        }
        auto it = std::find(active_features.begin(), active_features.end(), idx);
        if (it != active_features.end()) active_features.erase(it);
    }

    void recompute_from_position(const Position &p) {
        reset();
        active_features = position_features(p);
        for (int idx : active_features) {
            for (int j = 0; j < ACCUMULATOR_SIZE; ++j) {
                acc[j] += g_net.w1[idx][j];
            }
        }
    }

    // Differential update: add newly-appeared features, remove disappeared ones.
    void update_from_diff(const Position &prev, const Position &curr) {
        int added[8], removed[8], n_added = 0, n_removed = 0;
        position_diff(prev, curr, added, n_added, removed, n_removed);
        for (int i = 0; i < n_added; ++i) add_feature(added[i]);
        for (int i = 0; i < n_removed; ++i) remove_feature(removed[i]);
    }
};

static thread_local Accumulator g_acc;

// ---------------------------------------------------------------------------
// Forward pass (integer-only)
// ---------------------------------------------------------------------------
// Returns move logits for all 4096 (from,to) slots.  The caller must apply
// the legality mask.  Also fills `out_value` with the scalar position value.
// ---------------------------------------------------------------------------
static void forward_pass(const Position &p, int32_t out_logits[OUTPUT_SLOTS], int32_t &out_value) {
    g_acc.recompute_from_position(p);

    int16_t hidden[HIDDEN_SIZE];
    for (int k = 0; k < HIDDEN_SIZE; ++k) {
        int32_t sum = g_net.b2[k];

        // Platform-agnostic SIMD abstraction.  The same source compiles to
        // ARM NEON on this host and to AVX2/AVX-512 on x86_64 cloud nodes.
        // OpenEvolve should mutate only the bodies of these branches.
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        // ARM NEON code paths
        for (int j = 0; j < ACCUMULATOR_SIZE; ++j) {
            sum += (g_acc.acc[j] * g_net.w2[j][k]) >> QUANT_SHIFT;
        }
#elif defined(__AVX2__) || defined(__AVX512F__)
        // Accelerated Intel/AMD AVX code paths
        for (int j = 0; j < ACCUMULATOR_SIZE; ++j) {
            sum += (g_acc.acc[j] * g_net.w2[j][k]) >> QUANT_SHIFT;
        }
#else
        // Standard compiler-autovectorized fallback loops
        for (int j = 0; j < ACCUMULATOR_SIZE; ++j) {
            sum += (g_acc.acc[j] * g_net.w2[j][k]) >> QUANT_SHIFT;
        }
#endif
        hidden[k] = relu16(clamp16(sum));
    }

    // Output layer (move logits) + value head.
    for (int l = 0; l < OUTPUT_SLOTS; ++l) {
        int32_t sum = g_net.b3[l];
        for (int k = 0; k < HIDDEN_SIZE; ++k) {
            sum += (hidden[k] * g_net.w3[k][l]) >> QUANT_SHIFT;
        }
        out_logits[l] = sum;
    }

    int32_t val = g_net.bv;
    for (int k = 0; k < HIDDEN_SIZE; ++k) {
        val += (hidden[k] * g_net.wv[k]) >> QUANT_SHIFT;
    }
    out_value = val;
}

// ---------------------------------------------------------------------------
// Move selection with hard-coded legality mask
// ---------------------------------------------------------------------------
static Move select_masked_move(Position &p) {
    int32_t logits[OUTPUT_SLOTS];
    int32_t value;
    forward_pass(p, logits, value);

    auto moves = legal_moves(p);
    if (moves.empty()) return {-1, -1, -1};

    // Apply legality mask: every slot not corresponding to a legal move is
    // forced to a very negative score, leaving only legal moves selectable.
    int32_t masked[OUTPUT_SLOTS];
    for (int l = 0; l < OUTPUT_SLOTS; ++l) masked[l] = INT32_MIN;
    for (const auto &m : moves) {
        int slot = m.from * MOVE_STRIDE + m.to;
        if (slot >= 0 && slot < OUTPUT_SLOTS) {
            masked[slot] = logits[slot];
        }
    }

    // Argmax over the masked scores.
    int best_slot = 0;
    int32_t best_score = masked[0];
    for (int l = 1; l < OUTPUT_SLOTS; ++l) {
        if (masked[l] > best_score) {
            best_score = masked[l];
            best_slot = l;
        }
    }

    // Recover move.  If the best slot belongs to a promotion square, default
    // to queen promotion (promo index 3).
    int from = best_slot / MOVE_STRIDE;
    int to = best_slot % MOVE_STRIDE;
    int promo = -1;
    int rank8 = (p.side == WHITE) ? 7 : 0;
    if (to / 8 == rank8) {
        // Only pawn moves reach promotion; verify and choose queen.
        bool is_pawn = (p.bb[piece_index(p.side, PAWN)] & bit(from)) != 0;
        if (is_pawn) promo = 3;
    }
    return {from, to, promo};
}

// Choose a random legal move (used for opponent / baseline).
Move choose_move(Position &p) {
    auto moves = legal_moves(p);
    if (moves.empty()) return {-1, -1, -1};
    std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
    return moves[dist(g_rng)];
}

// Policy move: single forward pass through the DFCAM, masked by legal moves.
Move policy_move(Position &p) {
    return select_masked_move(p);
}

// ---------------------------------------------------------------------------
// Weight file I/O (binary flat dump of g_net)
// ---------------------------------------------------------------------------
static const char *WEIGHT_FILE_MAGIC = "AESTATEW";

bool save_weights(const char *path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(WEIGHT_FILE_MAGIC, 8);
    out.write(reinterpret_cast<const char*>(&g_net), sizeof(g_net));
    return out.good();
}

bool load_weights(const char *path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[8] = {};
    in.read(magic, 8);
    if (std::memcmp(magic, WEIGHT_FILE_MAGIC, 8) != 0) return false;
    // Accept variable-length weight payloads so the LLM can experiment with
    // new topologies without the evaluator hard-coding the exact struct size.
    // We copy as many bytes as fit into g_net and zero-pad the remainder.
    std::vector<char> buffer((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (buffer.empty()) return false;
    size_t to_copy = std::min(buffer.size(), sizeof(g_net));
    std::memcpy(&g_net, buffer.data(), to_copy);
    if (to_copy < sizeof(g_net)) {
        std::memset(reinterpret_cast<char*>(&g_net) + to_copy, 0, sizeof(g_net) - to_copy);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Self-play data generation mode for the Python training loop
// ---------------------------------------------------------------------------
// Each record is:
//   int32_t side_to_move;          // 0 = White, 1 = Black
//   int32_t num_features;
//   int32_t feature_indices[32];
//   int32_t chosen_from;
//   int32_t chosen_to;
//   int32_t chosen_promo;
//   int32_t outcome;               // final result from side_to_move perspective
//
// The training loop can reconstruct the position from feature_indices or
// simply feed them into its own network.
// ---------------------------------------------------------------------------
struct alignas(4) TrainingRecord {
    int32_t side_to_move;
    int32_t n_features;
    int32_t features[MAX_FEATURES];
    int32_t from;
    int32_t to;
    int32_t promo;
    int32_t outcome;
};

static void generate_training_data(int games, FILE *out) {
    for (int g = 0; g < games; ++g) {
        Position p = start_position();
        int ply = 0;
        int result = game_result(p, ply);
        std::vector<TrainingRecord> records;
        records.reserve(128);

        while (result == 2) {
            auto feats = position_features(p);
            TrainingRecord rec;
            rec.side_to_move = (int32_t)p.side;
            rec.n_features = (int32_t)std::min<size_t>(feats.size(), (size_t)MAX_FEATURES);
            for (size_t i = 0; i < (size_t)rec.n_features; ++i) rec.features[i] = feats[i];
            for (size_t i = (size_t)rec.n_features; i < (size_t)MAX_FEATURES; ++i) rec.features[i] = -1;

            Move m = policy_move(p);
            if (m.from < 0) break;

            rec.from = m.from;
            rec.to = m.to;
            rec.promo = m.promo;
            rec.outcome = 0; // filled in after game ends
            records.push_back(rec);

            make_move(p, m);
            ++ply;
            result = game_result(p, ply);
        }

        int white_outcome = 0;
        if (result == 1) white_outcome = 1;
        else if (result == -1) white_outcome = -1;

        for (auto &rec : records) {
            // outcome from the perspective of the side that made the recorded move
            int side_outcome = (rec.side_to_move == WHITE) ? white_outcome : -white_outcome;
            rec.outcome = side_outcome;
            std::fwrite(&rec, sizeof(rec), 1, out);
        }
    }
}

// # EVOLVE-BLOCK-END

// ---------------------------------------------------------------------------
// Entry point (fixed harness is in chess_runtime.hpp)
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    init_masks();
    init_net();

    std::string mode = (argc > 1) ? argv[1] : "bench";

    if (mode == "load_weights" && argc > 2) {
        bool ok = load_weights(argv[2]);
        if (!ok) {
            std::cerr << "Failed to load weights from " << argv[2] << std::endl;
            return 1;
        }
        std::cerr << "WEIGHTS loaded=" << argv[2] << std::endl;
        // Remaining arguments specify the mode to run with the loaded weights.
        // Special case: generate training data using the loaded weights.
        if (argc > 3 && std::string(argv[3]) == "generate_data") {
            int games = (argc > 4) ? std::stoi(argv[4]) : 100;
            generate_training_data(games, stdout);
            return 0;
        }
        if (argc > 3) {
            std::vector<char*> args;
            args.reserve(argc - 2);
            args.push_back(argv[0]);
            for (int i = 3; i < argc; ++i) args.push_back(argv[i]);
            return aetherstate_main((int)args.size(), args.data());
        }
        return 0;
    }

    if (mode == "generate_data") {
        int games = (argc > 2) ? std::stoi(argv[2]) : 100;
        generate_training_data(games, stdout);
        return 0;
    }

    return aetherstate_main(argc, argv);
}
'''


# ---------------------------------------------------------------------------
# Mutable Python training pipeline.
# Architecture constants are injected via an generated arch_constants.py file.
# ---------------------------------------------------------------------------
TRAIN_LOOP_PY = r'''#!/usr/bin/env python3
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
# Architecture constants (injected by the evaluator into arch_constants.py)
# ---------------------------------------------------------------------------
# The evaluator writes an arch_constants.py file next to this script before it
# is imported.  That file is the single source of truth for the architecture
# and must stay in sync with the C++ engine's injected constants.
try:
    from arch_constants import (
        INPUT_FEATURES,
        ACCUMULATOR_SIZE,
        HIDDEN_SIZE,
        OUTPUT_SLOTS,
        WEIGHT_MAGIC,
        TRAINING_RECORD_DTYPE,
    )
except ImportError:  # pragma: no cover
    # Fallback for running train_loop.py directly in legacy mode.  These values
    # must match the default ARCHITECTURE dict in aetherstate_bundle.py.
    if HAS_TORCH:
        INPUT_FEATURES = 768
        ACCUMULATOR_SIZE = 256
        HIDDEN_SIZE = 32
        OUTPUT_SLOTS = 4096
        WEIGHT_MAGIC = b"AESTATEW"

        _LEGACY_MAX_FEATURES = 32
        TRAINING_RECORD_DTYPE = np.dtype([
            ("side_to_move", "<i4"),
            ("n_features", "<i4"),
            ("features", "<i4", _LEGACY_MAX_FEATURES),
            ("from_", "<i4"),
            ("to_", "<i4"),
            ("promo", "<i4"),
            ("outcome", "<i4"),
        ])
    else:
        INPUT_FEATURES = ACCUMULATOR_SIZE = HIDDEN_SIZE = OUTPUT_SLOTS = 0
        WEIGHT_MAGIC = b""
        TRAINING_RECORD_DTYPE = None


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
    max_features = features.size(1)
    mask = torch.arange(max_features, device=device).unsqueeze(0) < n_features.unsqueeze(1)
    return features, actions, outcomes, mask


# ---------------------------------------------------------------------------
# Weight export (int8 / int16) for the C++ engine
# ---------------------------------------------------------------------------
def _pack_int8(arr: np.ndarray) -> bytes:
    return np.clip(np.rint(arr), -128, 127).astype(np.int8).tobytes()


def _pack_int16(arr: np.ndarray) -> bytes:
    return np.clip(np.rint(arr), -32768, 32767).astype(np.int16).tobytes()


def export_weights(model: nn.Module, path: Path) -> None:
    """Export trained PyTorch weights in the exact C++ NeuralNet layout.

    The C++ engine's NeuralNet struct is the source of truth for the expected
    weight size.  We compute the corresponding padded size from the injected
    architecture constants so the exported file can be loaded by the current
    engine.  If the LLM mutates the topology, it should update both the C++
    struct and this exporter consistently.
    """
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

    # Pad to the C++ struct size (multiple of 64).  The size is derived from
    # the injected architecture constants so the exporter stays in sync with the
    # C++ NeuralNet struct without hard-coding a global NET_SIZE.
    raw_net_size = (
        INPUT_FEATURES * ACCUMULATOR_SIZE * 1
        + ACCUMULATOR_SIZE * 2
        + ACCUMULATOR_SIZE * HIDDEN_SIZE * 1
        + HIDDEN_SIZE * 2
        + HIDDEN_SIZE * OUTPUT_SLOTS * 1
        + OUTPUT_SLOTS * 2
        + HIDDEN_SIZE * 1
        + 2
    )
    net_size = ((raw_net_size + 63) // 64) * 64
    if len(payload) < net_size:
        payload += b"\x00" * (net_size - len(payload))
    elif len(payload) > net_size:
        raise RuntimeError(f"Exported weights too large: {len(payload)} > {net_size}")

    with open(path, "wb") as f:
        f.write(WEIGHT_MAGIC)
        f.write(payload)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _features_to_input(batch_features: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert a batch of feature-index records into a dense multi-hot input tensor."""
    bsz = batch_features.size(0)
    x = torch.zeros(bsz, INPUT_FEATURES, device=device)
    if bsz > 0:
        valid = batch_features >= 0
        if valid.any():
            rows = torch.arange(bsz, device=device).unsqueeze(1).expand(-1, batch_features.size(1))[valid]
            cols = batch_features[valid]
            x[rows, cols] = 1.0
    return x


def _compute_batch_loss(
    model: nn.Module,
    features: torch.Tensor,
    actions: torch.Tensor,
    outcomes: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return the combined policy+value loss for a batch (no backward)."""
    device = next(model.parameters()).device
    x = _features_to_input(features, device)
    policy_logits, value = model(x)
    log_probs = F.log_softmax(policy_logits, dim=-1)
    bsz = actions.size(0)
    chosen_log_probs = log_probs[torch.arange(bsz, device=device), actions]
    policy_loss = -(chosen_log_probs * outcomes).mean()
    value_loss = F.mse_loss(value, outcomes)
    return policy_loss + 0.5 * value_loss


def compute_loss(
    model: nn.Module,
    features: torch.Tensor,
    actions: torch.Tensor,
    outcomes: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int = 64,
) -> float:
    """Compute the average loss over a dataset without training."""
    model.eval()
    n = features.size(0)
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_features = features[i : i + batch_size]
            batch_actions = actions[i : i + batch_size]
            batch_outcomes = outcomes[i : i + batch_size]
            batch_mask = mask[i : i + batch_size]
            loss = _compute_batch_loss(model, batch_features, batch_actions, batch_outcomes, batch_mask)
            total_loss += loss.item() * batch_features.size(0)
    model.train()
    return total_loss / max(n, 1)


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
            batch_actions = actions[idx]
            batch_outcomes = outcomes[idx]

            x = _features_to_input(batch_features, device)
            policy_logits, value = model(x)

            # REINFORCE policy loss: maximize log-prob of chosen action weighted
            # by the final outcome from the acting side's perspective.
            log_probs = F.log_softmax(policy_logits, dim=-1)
            bsz = batch_features.size(0)
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
    parser.add_argument("--weights-path", type=Path, default=None, help="Optional explicit path for exported weights.  If omitted, a temporary path is used.")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to a fixed binary dataset.  If provided, self-play generation is skipped and this data is used for train/val.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of the dataset to reserve for validation (default 0.2).")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap the number of training samples used (default: use all).")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Cap the number of validation samples used (default: use all).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible training/benchmarking. If None, no explicit seeding is done.")
    args = parser.parse_args(argv)

    # Set random seeds for reproducible evaluation if requested.
    if args.seed is not None:
        import random as _random
        _random.seed(args.seed)
        if HAS_TORCH:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

    source_path = args.source_path.resolve()
    if not source_path.exists():
        print(json.dumps({"error": f"source not found: {source_path}", "fitness": 0.0}))
        return 1

    work_dir = Path(tempfile.mkdtemp(prefix="aetherstate_"))
    binary_path = work_dir / "aetherstate"
    weights_path = args.weights_path or (work_dir / "weights.bin")
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

    # 2) Load or generate training data.
    if args.dataset is not None:
        data_path = args.dataset.resolve()
        if not data_path.exists():
            result["error"] = f"dataset not found: {data_path}"
            print(_format(result, args.output))
            return 1
    else:
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

    # 4) Split train/val and train.
    n = len(records)
    val_count = max(1, int(n * args.val_split))
    val_count = min(val_count, n // 2)
    train_count = n - val_count

    train_features, val_features = features[:train_count], features[train_count:]
    train_actions, val_actions = actions[:train_count], actions[train_count:]
    train_outcomes, val_outcomes = outcomes[:train_count], outcomes[train_count:]
    train_mask, val_mask = mask[:train_count], mask[train_count:]

    # Cap sample counts for fast micro-evaluation.
    if args.max_train_samples is not None:
        train_features = train_features[: args.max_train_samples]
        train_actions = train_actions[: args.max_train_samples]
        train_outcomes = train_outcomes[: args.max_train_samples]
        train_mask = train_mask[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_features = val_features[: args.max_val_samples]
        val_actions = val_actions[: args.max_val_samples]
        val_outcomes = val_outcomes[: args.max_val_samples]
        val_mask = val_mask[: args.max_val_samples]

    model = AetherStateNet().to(device)
    train_model(
        model,
        train_features,
        train_actions,
        train_outcomes,
        train_mask,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    val_loss = compute_loss(model, val_features, val_actions, val_outcomes, val_mask)
    result["val_loss"] = val_loss

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
    draw_rate = conv.get("draw_rate", 0.0)
    avg_game_length = conv.get("avg_game_length", 0.0)
    # Fitness used for selection; MAP-Elites axes are reported separately.
    fitness = (win_rate * 1000.0) + (nodes_per_second / 100000.0)
    out = {
        "nodes_per_second": nodes_per_second,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "avg_game_length": avg_game_length,
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
'''
