/*
 * AetherState Chess Runtime
 * -------------------------
 * Fixed, non-learned chess engine infrastructure.  This header is included by
 * seed_engine.cpp and is intentionally kept OUT of the OpenEvolve mutation
 * pipeline.  It provides the hard-coded chess rule mask, move generator,
 * benchmark harness, and the executable entry point.
 *
 * SIMD usage inside the mutable seed is wrapped with preprocessor macros so
 * the same source compiles with ARM NEON (this host) and AVX2/AVX-512 (rented
 * x86_64 cloud nodes).  This header defines helper macros used by the NN
 * block in seed_engine.cpp.
 *
 * Build: g++ -std=c++17 -O3 seed_engine.cpp -o aetherstate
 */

#ifndef AETHERSTATE_CHESS_RUNTIME_HPP
#define AETHERSTATE_CHESS_RUNTIME_HPP

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Platform-agnostic SIMD abstraction macros
// ---------------------------------------------------------------------------

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define AETHER_SIMD_NEON
#elif defined(__AVX2__) || defined(__AVX512F__)
    #include <immintrin.h>
    #define AETHER_SIMD_AVX
#else
    #define AETHER_SIMD_SCALAR
#endif

// Dot-product helper used by the mutable NN block.  It multiplies `n`
// 16-bit values in `a` by 16-bit values in `b` and returns a 32-bit sum.
// The implementation is selected at compile time; the seed block may call
// this macro inside loops or unroll manually.
#if defined(AETHER_SIMD_NEON)
    inline int32_t simd_dot16(const int16_t* a, const int16_t* b, int n) {
        int32x4_t acc = vdupq_n_s32(0);
        for (int i = 0; i < n; i += 8) {
            int16x8_t va = vld1q_s16(a + i);
            int16x8_t vb = vld1q_s16(b + i);
            int32x4_t lo = vmull_s16(vget_low_s16(va), vget_low_s16(vb));
            int32x4_t hi = vmull_s16(vget_high_s16(va), vget_high_s16(vb));
            acc = vaddq_s32(acc, vaddq_s32(lo, hi));
        }
        return vaddvq_s32(acc);
    }
    #define SIMD_DOT16(a, b, n) simd_dot16((a), (b), (n))
#elif defined(AETHER_SIMD_AVX)
    inline int32_t simd_dot16(const int16_t* a, const int16_t* b, int n) {
        __m256i acc = _mm256_setzero_si256();
        for (int i = 0; i < n; i += 16) {
            __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
            __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
            __m256i p = _mm256_madd_epi16(va, vb);
            acc = _mm256_add_epi32(acc, p);
        }
        __m128i lo = _mm256_castsi256_si128(acc);
        __m128i hi = _mm256_extracti128_si256(acc, 1);
        __m128i s = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        return _mm_cvtsi128_si32(s);
    }
    #define SIMD_DOT16(a, b, n) simd_dot16((a), (b), (n))
#else
    inline int32_t simd_dot16(const int16_t* a, const int16_t* b, int n) {
        int32_t sum = 0;
        for (int i = 0; i < n; ++i) sum += a[i] * b[i];
        return sum;
    }
    #define SIMD_DOT16(a, b, n) simd_dot16((a), (b), (n))
#endif

// ---------------------------------------------------------------------------
// Fixed bitboard / chess rule infrastructure
// ---------------------------------------------------------------------------

using U64 = uint64_t;

enum PieceType { PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_NB };
enum Color { WHITE = 0, BLACK = 1, COLOR_NB };

static inline int piece_index(Color c, PieceType pt) { return c * 6 + pt; }

static inline U64 bit(int sq) { return U64(1) << sq; }
static inline int popcount(U64 b) { return __builtin_popcountll(b); }
static inline int lsb(U64 b) { return __builtin_ctzll(b); }
static inline U64 pop_lsb(U64 &b) {
    int s = lsb(b);
    b &= b - 1;
    return bit(s);
}

static const int FILE_A = 0, FILE_H = 7;
static const int RANK_1 = 0, RANK_8 = 7;

static inline bool on_board(int file, int rank) {
    return file >= 0 && file <= 7 && rank >= 0 && rank <= 7;
}

static U64 knight_attacks[64];
static U64 king_attacks[64];
static U64 rank_mask[64];
static U64 file_mask[64];
static U64 diag_mask[64];
static U64 anti_mask[64];

static void init_masks() {
    for (int sq = 0; sq < 64; ++sq) {
        int f = sq % 8, r = sq / 8;
        knight_attacks[sq] = 0;
        const int kdx[8] = {-2, -1, 1, 2, 2, 1, -1, -2};
        const int kdy[8] = {1, 2, 2, 1, -1, -2, -2, -1};
        for (int i = 0; i < 8; ++i) {
            int nf = f + kdx[i], nr = r + kdy[i];
            if (on_board(nf, nr)) knight_attacks[sq] |= bit(nr * 8 + nf);
        }
        king_attacks[sq] = 0;
        for (int df = -1; df <= 1; ++df)
            for (int dr = -1; dr <= 1; ++dr) {
                if (df == 0 && dr == 0) continue;
                int nf = f + df, nr = r + dr;
                if (on_board(nf, nr)) king_attacks[sq] |= bit(nr * 8 + nf);
            }
        rank_mask[sq] = 0xFFULL << (r * 8);
        file_mask[sq] = 0x0101010101010101ULL << f;
        diag_mask[sq] = 0;
        anti_mask[sq] = 0;
        for (int nf = 0; nf < 8; ++nf) {
            int nr = nf - f + r;
            if (on_board(nf, nr)) diag_mask[sq] |= bit(nr * 8 + nf);
            int nr2 = f + r - nf;
            if (on_board(nf, nr2)) anti_mask[sq] |= bit(nr2 * 8 + nf);
        }
    }
}

struct Position {
    U64 bb[12];
    Color side;
    int castling;
    int ep;
    int halfmove;
    int fullmove;
};

static const int CASTLE_WK = 1, CASTLE_WQ = 2, CASTLE_BK = 4, CASTLE_BQ = 8;

static Position start_position() {
    Position p;
    memset(&p, 0, sizeof(p));
    p.bb[piece_index(WHITE, PAWN)]   = 0x000000000000FF00ULL;
    p.bb[piece_index(WHITE, KNIGHT)] = 0x0000000000000042ULL;
    p.bb[piece_index(WHITE, BISHOP)] = 0x0000000000000024ULL;
    p.bb[piece_index(WHITE, ROOK)]   = 0x0000000000000081ULL;
    p.bb[piece_index(WHITE, QUEEN)]  = 0x0000000000000008ULL;
    p.bb[piece_index(WHITE, KING)]   = 0x0000000000000010ULL;
    p.bb[piece_index(BLACK, PAWN)]   = 0x00FF000000000000ULL;
    p.bb[piece_index(BLACK, KNIGHT)] = 0x4200000000000000ULL;
    p.bb[piece_index(BLACK, BISHOP)] = 0x2400000000000000ULL;
    p.bb[piece_index(BLACK, ROOK)]   = 0x8100000000000000ULL;
    p.bb[piece_index(BLACK, QUEEN)]  = 0x0800000000000000ULL;
    p.bb[piece_index(BLACK, KING)]   = 0x1000000000000000ULL;
    p.side = WHITE;
    p.castling = CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ;
    p.ep = -1;
    p.halfmove = 0;
    p.fullmove = 1;
    return p;
}

static U64 occupied(const Position &p) {
    U64 o = 0;
    for (int i = 0; i < 12; ++i) o |= p.bb[i];
    return o;
}

static U64 color_occ(const Position &p, Color c) {
    U64 o = 0;
    for (int pt = 0; pt < 6; ++pt) o |= p.bb[piece_index(c, (PieceType)pt)];
    return o;
}

static U64 attacks_by(const Position &p, Color c) {
    U64 occ = occupied(p);
    U64 att = 0;
    U64 pawns = p.bb[piece_index(c, PAWN)];
    if (c == WHITE) {
        att |= ((pawns & ~0x8080808080808080ULL) << 7);
        att |= ((pawns & ~0x0101010101010101ULL) << 9);
    } else {
        att |= ((pawns & ~0x8080808080808080ULL) >> 9);
        att |= ((pawns & ~0x0101010101010101ULL) >> 7);
    }
    U64 kn = p.bb[piece_index(c, KNIGHT)];
    while (kn) { int sq = lsb(kn); kn &= kn - 1; att |= knight_attacks[sq]; }
    U64 k = p.bb[piece_index(c, KING)];
    if (k) { int sq = lsb(k); att |= king_attacks[sq]; }
    U64 bishops = p.bb[piece_index(c, BISHOP)] | p.bb[piece_index(c, QUEEN)];
    U64 rooks   = p.bb[piece_index(c, ROOK)]   | p.bb[piece_index(c, QUEEN)];
    while (bishops) {
        int sq = lsb(bishops); bishops &= bishops - 1;
        U64 blockers = occ & diag_mask[sq];
        U64 attacks = 0;
        U64 b = blockers;
        while (b) { int bs = lsb(b); b &= b - 1; attacks |= (diag_mask[sq] & ~diag_mask[bs] & ~bit(bs)); }
        if (!blockers) attacks = diag_mask[sq];
        att |= attacks & diag_mask[sq];
    }
    while (rooks) {
        int sq = lsb(rooks); rooks &= rooks - 1;
        U64 blockers = occ & (rank_mask[sq] | file_mask[sq]);
        U64 attacks = 0;
        U64 b = blockers;
        while (b) { int bs = lsb(b); b &= b - 1;
            attacks |= ((rank_mask[sq] | file_mask[sq]) & ~rank_mask[bs] & ~file_mask[bs] & ~bit(bs));
        }
        if (!blockers) attacks = (rank_mask[sq] | file_mask[sq]);
        att |= attacks & (rank_mask[sq] | file_mask[sq]);
    }
    return att;
}

static bool in_check(const Position &p, Color c) {
    U64 king = p.bb[piece_index(c, KING)];
    if (!king) return false;
    int sq = lsb(king);
    return (attacks_by(p, (Color)(c ^ 1)) & bit(sq)) != 0;
}

struct Move {
    int from, to, promo;
};

static void make_move(Position &p, const Move &m) {
    Color us = p.side, them = (Color)(us ^ 1);
    int moved_piece = -1;
    for (int pt = 0; pt < 6; ++pt) {
        if (p.bb[piece_index(us, (PieceType)pt)] & bit(m.from)) { moved_piece = pt; break; }
    }
    if (moved_piece < 0) return;
    p.bb[piece_index(us, (PieceType)moved_piece)] &= ~bit(m.from);
    for (int pt = 0; pt < 6; ++pt)
        p.bb[piece_index(them, (PieceType)pt)] &= ~bit(m.to);
    if (moved_piece == PAWN && (m.to / 8 == 0 || m.to / 8 == 7)) {
        p.bb[piece_index(us, (PieceType)(m.promo < 0 ? QUEEN : m.promo == 0 ? KNIGHT : m.promo == 1 ? BISHOP : m.promo == 2 ? ROOK : QUEEN))] |= bit(m.to);
    } else {
        p.bb[piece_index(us, (PieceType)moved_piece)] |= bit(m.to);
    }
    if (m.from == 4) p.castling &= us == WHITE ? ~(CASTLE_WK | CASTLE_WQ) : ~(CASTLE_BK | CASTLE_BQ);
    if (m.from == 0 || m.to == 0) p.castling &= ~CASTLE_WQ;
    if (m.from == 7 || m.to == 7) p.castling &= ~CASTLE_WK;
    if (m.from == 56 || m.to == 56) p.castling &= ~CASTLE_BQ;
    if (m.from == 63 || m.to == 63) p.castling &= ~CASTLE_BK;
    if (moved_piece == KING && std::abs(m.to - m.from) == 2) {
        if (m.to > m.from) {
            p.bb[piece_index(us, ROOK)] &= ~bit(m.to + 1);
            p.bb[piece_index(us, ROOK)] |= bit(m.to - 1);
        } else {
            p.bb[piece_index(us, ROOK)] &= ~bit(m.to - 2);
            p.bb[piece_index(us, ROOK)] |= bit(m.to + 1);
        }
    }
    if (moved_piece == PAWN && m.to == p.ep) {
        int cap_sq = us == WHITE ? m.to - 8 : m.to + 8;
        p.bb[piece_index(them, PAWN)] &= ~bit(cap_sq);
    }
    if (moved_piece == PAWN && std::abs(m.to - m.from) == 16) {
        p.ep = (m.from + m.to) / 2;
    } else {
        p.ep = -1;
    }
    p.halfmove = (moved_piece == PAWN) ? 0 : p.halfmove + 1;
    if (us == BLACK) ++p.fullmove;
    p.side = them;
}

static std::vector<Move> legal_moves(const Position &p) {
    std::vector<Move> moves;
    Color us = p.side, them = (Color)(us ^ 1);
    U64 occ = occupied(p);
    U64 not_us = ~color_occ(p, us);
    U64 pawns = p.bb[piece_index(us, PAWN)];
    U64 promo_rank = us == WHITE ? 0xFF00000000000000ULL : 0x00000000000000FFULL;
    for (int sq = 0; sq < 64; ++sq) {
        if (!(pawns & bit(sq))) continue;
        int f = sq % 8, r = sq / 8;
        int dir = us == WHITE ? 1 : -1;
        int nr = r + dir;
        if (nr < 0 || nr > 7) continue;
        int forward = nr * 8 + f;
        if (!(occ & bit(forward))) {
            if (promo_rank & bit(forward)) {
                for (int pr = 0; pr < 4; ++pr) moves.push_back({sq, forward, pr});
            } else {
                moves.push_back({sq, forward, -1});
            }
        }
        if (((us == WHITE && r == 1) || (us == BLACK && r == 6)) && !(occ & bit(forward)) && !(occ & bit(forward + dir * 8))) {
            moves.push_back({sq, forward + dir * 8, -1});
        }
        for (int df = -1; df <= 1; df += 2) {
            int nf = f + df;
            if (nf < 0 || nf > 7) continue;
            int cap = nr * 8 + nf;
            bool cap_piece = (color_occ(p, them) & bit(cap)) != 0;
            bool ep = (p.ep == cap);
            if (cap_piece || ep) {
                if (promo_rank & bit(cap)) {
                    for (int pr = 0; pr < 4; ++pr) moves.push_back({sq, cap, pr});
                } else {
                    moves.push_back({sq, cap, -1});
                }
            }
        }
    }
    U64 kn = p.bb[piece_index(us, KNIGHT)];
    while (kn) { int sq = lsb(kn); kn &= kn - 1;
        U64 at = knight_attacks[sq] & not_us;
        while (at) { int tsq = lsb(at); at &= at - 1; moves.push_back({sq, tsq, -1}); }
    }
    U64 k = p.bb[piece_index(us, KING)];
    if (k) {
        int sq = lsb(k);
        U64 at = king_attacks[sq] & not_us;
        while (at) { int tsq = lsb(at); at &= at - 1; moves.push_back({sq, tsq, -1}); }
        int rank_base = us == WHITE ? 0 : 56;
        U64 safe = ~attacks_by(p, them);
        if (us == WHITE) {
            if ((p.castling & CASTLE_WK) && !(occ & (0x60ULL << rank_base)) && ((safe & (0x70ULL << rank_base)) == (0x70ULL & (safe & (0x70ULL << rank_base)))))
                moves.push_back({sq, rank_base + 6, -1});
            if ((p.castling & CASTLE_WQ) && !(occ & (0x0EULL << rank_base)) && ((safe & (0x1CULL << rank_base)) == (0x1CULL & (safe & (0x1CULL << rank_base)))))
                moves.push_back({sq, rank_base + 2, -1});
        } else {
            if ((p.castling & CASTLE_BK) && !(occ & (0x60ULL << (rank_base - 56))) && ((safe & (0x70ULL << (rank_base - 56))) == (0x70ULL & (safe & (0x70ULL << (rank_base - 56))))))
                moves.push_back({sq, rank_base + 6, -1});
            if ((p.castling & CASTLE_BQ) && !(occ & (0x0EULL << (rank_base - 56))) && ((safe & (0x1CULL << (rank_base - 56))) == (0x1CULL & (safe & (0x1CULL << (rank_base - 56))))))
                moves.push_back({sq, rank_base + 2, -1});
        }
    }
    U64 bishops = p.bb[piece_index(us, BISHOP)] | p.bb[piece_index(us, QUEEN)];
    U64 rooks   = p.bb[piece_index(us, ROOK)]   | p.bb[piece_index(us, QUEEN)];
    while (bishops) {
        int sq = lsb(bishops); bishops &= bishops - 1;
        for (int df = -1; df <= 1; df += 2) for (int dr = -1; dr <= 1; dr += 2) {
            for (int d = 1; d < 8; ++d) {
                int f = sq % 8 + df * d, r = sq / 8 + dr * d;
                if (!on_board(f, r)) break;
                int tsq = r * 8 + f; moves.push_back({sq, tsq, -1});
                if (occ & bit(tsq)) break;
            }
        }
    }
    while (rooks) {
        int sq = lsb(rooks); rooks &= rooks - 1;
        for (int dir = 0; dir < 4; ++dir) {
            const int df[4] = {1,-1,0,0}, dr[4] = {0,0,1,-1};
            for (int d = 1; d < 8; ++d) {
                int f = sq % 8 + df[dir] * d, r = sq / 8 + dr[dir] * d;
                if (!on_board(f, r)) break;
                int tsq = r * 8 + f; moves.push_back({sq, tsq, -1});
                if (occ & bit(tsq)) break;
            }
        }
    }
    std::vector<Move> legal;
    for (const auto &m : moves) {
        Position t = p;
        make_move(t, m);
        if (!in_check(t, us)) legal.push_back(m);
    }
    return legal;
}

// ---------------------------------------------------------------------------
// Forward declarations for the mutable AI code implemented in seed_engine.cpp
// ---------------------------------------------------------------------------

struct NeuralNet;

extern NeuralNet g_net;
extern std::mt19937_64 g_rng;

void init_net();
Move choose_move(Position &p);
Move policy_move(Position &p);

// Helper: return a list of (piece_index, square) changes between two positions.
// `out_added` and `out_removed` are cleared and filled with feature indices
// in [0, INPUT_FEATURES) where INPUT_FEATURES = 12 * 64.
template<int MAX_CHANGES = 8>
static void position_diff(const Position &prev, const Position &curr,
                            int *out_added, int &out_added_count,
                            int *out_removed, int &out_removed_count) {
    out_added_count = 0;
    out_removed_count = 0;
    for (int pt = 0; pt < 12; ++pt) {
        U64 diff = prev.bb[pt] ^ curr.bb[pt];
        while (diff) {
            int sq = lsb(diff);
            diff &= diff - 1;
            int idx = (pt * 64) + sq;
            bool in_prev = (prev.bb[pt] & bit(sq)) != 0;
            bool in_curr = (curr.bb[pt] & bit(sq)) != 0;
            if (in_prev && !in_curr && out_removed_count < MAX_CHANGES) {
                out_removed[out_removed_count++] = idx;
            } else if (!in_prev && in_curr && out_added_count < MAX_CHANGES) {
                out_added[out_added_count++] = idx;
            }
        }
    }
}

// Convert a Position into a sparse list of active feature indices.
static std::vector<int> position_features(const Position &p) {
    std::vector<int> feats;
    feats.reserve(32);
    for (int pt = 0; pt < 12; ++pt) {
        U64 b = p.bb[pt];
        while (b) {
            int sq = lsb(b);
            b &= b - 1;
            feats.push_back(pt * 64 + sq);
        }
    }
    return feats;
}

// ---------------------------------------------------------------------------
// Fixed benchmark / game harness
// ---------------------------------------------------------------------------

static int game_result(Position &p, int ply) {
    if (p.halfmove >= 100) return 0;
    auto moves = legal_moves(p);
    if (moves.empty()) return in_check(p, p.side) ? (p.side == WHITE ? -1 : 1) : 0;
    if (ply > 500) return 0;
    return 2;
}

static int play_random_game(bool vs_random) {
    Position p = start_position();
    int ply = 0;
    int result = game_result(p, ply);
    while (result == 2) {
        Move m;
        if (!vs_random || p.side == WHITE) {
            m = policy_move(p);
        } else {
            m = choose_move(p);
        }
        if (m.from < 0) break;
        make_move(p, m);
        ++ply;
        result = game_result(p, ply);
    }
    return result;
}

static double run_benchmark(long long steps) {
    Position p = start_position();
    auto start = std::chrono::high_resolution_clock::now();
    long long nodes = 0;
    for (long long i = 0; i < steps; ++i) {
        auto moves = legal_moves(p);
        nodes += (long long)moves.size();
        if (!moves.empty()) {
            make_move(p, moves[i % moves.size()]);
        }
        if (moves.size() <= 1) p = start_position();
    }
    auto end = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(end - start).count();
    return (double)nodes / std::max(sec, 1e-9);
}

static double benchmark_seconds(long long steps, double max_seconds) {
    auto start = std::chrono::high_resolution_clock::now();
    long long nodes = 0;
    Position p = start_position();
    while (true) {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();
        if (elapsed >= max_seconds) break;
        for (long long i = 0; i < 1000; ++i) {
            auto moves = legal_moves(p);
            nodes += (long long)moves.size();
            if (!moves.empty()) make_move(p, moves[0]);
            if (moves.size() <= 1) p = start_position();
        }
    }
    double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    return (double)nodes / std::max(sec, 1e-9);
}

static double selfplay_winrate(int games) {
    int wins = 0, draws = 0;
    for (int i = 0; i < games; ++i) {
        int r = play_random_game(false);
        if (r == 1) ++wins; else if (r == 0) ++draws;
    }
    return (wins + 0.5 * draws) / games;
}

static double random_winrate(int games) {
    int wins = 0, draws = 0, losses = 0;
    for (int i = 0; i < games; ++i) {
        int r = play_random_game(true);
        if (r == 1) ++wins; else if (r == 0) ++draws; else ++losses;
    }
    return (wins + 0.5 * draws) / games;
}

// ---------------------------------------------------------------------------
// Inline executable entry point
// ---------------------------------------------------------------------------

inline int aetherstate_main(int argc, char **argv) {
    init_masks();
    init_net();

    std::string mode = (argc > 1) ? argv[1] : "bench";
    if (mode == "bench") {
        long long steps = (argc > 2) ? std::stoll(argv[2]) : 5000;
        double nps = run_benchmark(steps);
        std::cout << "BENCHMARK nodes_per_second=" << (long long)nps << std::endl;
        return 0;
    } else if (mode == "selfplay") {
        int games = (argc > 2) ? std::stoi(argv[2]) : 20;
        double wr = selfplay_winrate(games);
        std::cout << "SELFPLAY win_rate=" << wr << " games=" << games << std::endl;
        return 0;
    } else if (mode == "random") {
        int games = (argc > 2) ? std::stoi(argv[2]) : 20;
        double wr = random_winrate(games);
        std::cout << "RANDOM win_rate=" << wr << " games=" << games << std::endl;
        return 0;
    } else if (mode == "bench_time") {
        double sec = (argc > 2) ? std::stod(argv[2]) : 90.0;
        double nps = benchmark_seconds(5000, sec);
        std::cout << "BENCHMARK nodes_per_second=" << (long long)nps << std::endl;
        return 0;
    } else if (mode == "random_time") {
        double sec = (argc > 2) ? std::stod(argv[2]) : 180.0;
        auto start = std::chrono::high_resolution_clock::now();
        int wins = 0, draws = 0, losses = 0, games = 0;
        while (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count() < sec) {
            int r = play_random_game(true);
            if (r == 1) ++wins; else if (r == 0) ++draws; else ++losses;
            ++games;
        }
        double wr = games ? (wins + 0.5 * draws) / games : 0.0;
        std::cout << "RANDOM win_rate=" << wr << " games=" << games << std::endl;
        return 0;
    } else if (mode == "selfplay_time") {
        double sec = (argc > 2) ? std::stod(argv[2]) : 180.0;
        auto start = std::chrono::high_resolution_clock::now();
        int wins = 0, draws = 0, games = 0;
        while (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count() < sec) {
            int r = play_random_game(false);
            if (r == 1) ++wins; else if (r == 0) ++draws;
            ++games;
        }
        double wr = games ? (wins + 0.5 * draws) / games : 0.0;
        std::cout << "SELFPLAY win_rate=" << wr << " games=" << games << std::endl;
        return 0;
    } else if (mode == "micro") {
        long long steps = (argc > 2) ? std::stoll(argv[2]) : 5000;
        double nps = run_benchmark(steps);
        double wr = random_winrate(20);
        std::cout << "MICRO nodes_per_second=" << (long long)nps << " win_rate=" << wr << std::endl;
        return 0;
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }
}

#endif // AETHERSTATE_CHESS_RUNTIME_HPP
