/*
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
