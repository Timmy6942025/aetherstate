# AetherState

**Autonomous evolutionary research toward a zero-knowledge pure-AI chess engine.**

AetherState uses [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) — an LLM-driven evolutionary coding framework — to iteratively mutate and improve a C++ chess engine's neural network architecture, SIMD inference code, and PyTorch training pipeline. The goal is to produce the strongest possible engine code, which is then trained in one large final run on multi-GPU cloud hardware.

---

## Architecture

### DFCAM — Deep Fully-Connected Accumulator Matrix

```
Input:  768 sparse bitboard features (12 pieces × 64 squares)
  ↓
Layer 1: Accumulator(256)  — differential sparse add/remove updates
  ↓
Layer 2: Hidden(32)        — ReLU activation
  ↓
Output:  Move logits(4096) — masked by legal moves, argmax selected
Value:   Single scalar from the same trunk
```

- **Quantization:** int8 weights, int32 activations, fixed-point shift=7
- **SIMD:** Platform-agnostic preprocessor blocks for ARM NEON (local Pi) and AVX2/AVX-512 (cloud x86_64)

### File Structure

```
tasks/aetherstate/
├── seed_engine.cpp       # Mutable C++ inference core (OpenEvolve mutates this)
├── chess_runtime.hpp     # Protected chess rules, move generation, CLI harness
├── train_loop.py         # PyTorch RL training pipeline (REINFORCE, weight export)
├── evaluate.py           # Evaluator: compile → trainability gate → benchmarks
├── config.yaml           # OpenEvolve configuration (LLM, MAP-Elites, prompts)
└── outputs/              # Evolution outputs (best programs, logs, checkpoints)
```

### How Evolution Works

1. **OpenEvolve** takes `seed_engine.cpp` as the initial program
2. The **mutation LLM** (InclusionAI Ling 3.0 Flash via Novita AI) reads the current code and emits SEARCH/REPLACE diffs targeting only the `// # EVOLVE-BLOCK-START` … `// # EVOLVE-BLOCK-END` region
3. **evaluate.py** scores each mutant:
   - Compiles with `g++ -O3` (45s timeout)
   - Runs a **trainability gate** (generates 1 self-play game, trains 1 epoch in PyTorch, exports weights, loads them in C++)
   - Benchmarks inference speed (`nodes_per_second`)
   - Benchmarks win rate against a random legal-move opponent
4. **MAP-Elites** maintains a diverse population across two feature dimensions: speed and strength
5. The best programs survive; the rest are discarded

### What the LLM Can Mutate

Within the evolvable block, the LLM can improve:

- **SIMD/NEON/AVX intrinsics** for the DFCAM dot-product hot loops
- **Neural network layer sizes, weight layouts, activation functions, quantization**
- **Search routines** (alpha-beta depth, move ordering, iterative deepening, quiescence)
- **Self-play / training logic** (policy selection, reward shaping, gradient steps)

The LLM **cannot** touch the chess rules engine, legal move generator, or CLI harness — those are fixed in `chess_runtime.hpp`.

---

## Quick Start

### Prerequisites

- Python 3.10+ (tested on 3.13)
- `g++` with C++17 support
- A Novita AI API key (or any OpenAI-compatible provider)

### Setup

```bash
# Clone the repo
git clone https://github.com/Timmy6942025/aetherstate.git
cd aetherstate

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Set your API key
export NOVITA_API_KEY="sk_..."
```

### Run Evolution

```bash
cd openevolve_aetherstate
source .venv/bin/activate
export NOVITA_API_KEY="sk_..."

python openevolve-run.py \
  tasks/aetherstate/seed_engine.cpp \
  tasks/aetherstate/evaluate.py \
  --config tasks/aetherstate/config.yaml \
  --output tasks/aetherstate/outputs \
  --iterations 3
```

### Monitor Progress

```bash
# Watch the latest log
tail -f tasks/aetherstate/outputs/logs/openevolve_*.log

# Check the best program
cat tasks/aetherstate/outputs/best/best_program_info.json
```

---

## Configuration

The main configuration is in `tasks/aetherstate/config.yaml`. Key settings:

| Setting | Value | Description |
|---------|-------|-------------|
| `llm.models[0].name` | `inclusionai/ling-3.0-flash` | Mutation LLM (124B MoE, 262K context) |
| `llm.max_tokens` | 32768 | Max output tokens per mutation |
| `llm.temperature` | 0.7 | Balances exploration vs format compliance |
| `database.num_islands` | 3 | Parallel evolution populations |
| `database.feature_dimensions` | `[nodes_per_second, win_rate]` | MAP-Elites axes |
| `evaluator.timeout` | 600s | Max time per evaluation |
| `max_iterations` | 50 | Total evolution generations |

### LLM Provider

AetherState uses Novita AI's OpenAI-compatible API:

```yaml
llm:
  api_base: "https://api.novita.ai/openai"
  api_key: "${NOVITA_API_KEY}"
  models:
    - name: "inclusionai/ling-3.0-flash"
      weight: 1.0
```

You can swap to any OpenAI-compatible provider by changing `api_base`, `api_key`, and `models[0].name`.

---

## Platform Support

| Platform | SIMD Target | Compile Flag | Use Case |
|----------|-------------|--------------|----------|
| ARM64 (Raspberry Pi 4) | NEON | `-O3` | Local development & evolution |
| x86_64 (Cloud GPU nodes) | AVX2 / AVX-512 | `-O3 -mavx2` or `-mavx512f` | Final training run |

The code uses conditional preprocessor blocks to compile the optimal SIMD path for each platform:

```cpp
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // ARM NEON intrinsics
#elif defined(__AVX2__) || defined(__AVX512F__)
    // AVX2/AVX-512 intrinsics
#else
    // Compiler-autovectorized fallback
#endif
```

---

## Training Pipeline

### Trainability Gate (During Evolution)

The evaluator runs a lightweight gate before expensive benchmarks:

1. Compiles the mutated C++ code
2. Generates 1 self-play game of training data
3. Runs 1 epoch of REINFORCE training in PyTorch (CPU-only)
4. Exports quantized int8 weights to binary
5. Loads the weights back into the C++ engine and runs a forward pass

This catches structural bugs (shape mismatches, NaN, weight format errors) without doing real training.

### Final Training Run (Cloud GPUs)

After evolution finishes, take the best-evolved code and run one large training run:

```bash
# Generate millions of self-play games
./aetherstate generate_data 1000000 > training_data.bin

# Train with PyTorch on GPU
python train_loop.py --data training_data.bin --epochs 100 --batch-size 256 --lr 0.001

# Export final weights
# (train_loop.py exports weights.bin automatically)

# Run the final engine
./aetherstate load_weights weights.bin bench_time 60
```

---

## Evaluation Metrics

| Metric | Description | Evolution Target |
|--------|-------------|------------------|
| `nodes_per_second` | Inference speed (forward passes/sec) | Maximize |
| `win_rate` | Win rate vs random legal-move opponent | Maximize |
| `compile_ok` | Did the code compile? | Must be true |
| `trainable` | Did the trainability gate pass? | Must be true |
| `compile_time` | Compilation duration (seconds) | Minimize |
| `fitness` | Combined score for MAP-Elites | Maximize |

---

## Project Goals

1. **Iterative code improvement** — The evolutionary loop discovers better architectures, training algorithms, and inference code over many generations
2. **One big training run** — All evolution is preparation for a single, expensive training run on multi-GPU cloud hardware
3. **Zero-knowledge** — The engine learns chess entirely from self-play, with no human-provided chess knowledge beyond the rules
4. **Platform-agnostic** — Code compiles and runs on both ARM64 (local development) and x86_64 (cloud deployment)

---

## Development

### Running Tests

```bash
# Smoke test the evaluator
cd openevolve_aetherstate
source .venv/bin/activate
AETHERSTATE_BENCH_SECONDS=3 AETHERSTATE_RANDOM_SECONDS=5 \
  python tasks/aetherstate/evaluate.py tasks/aetherstate/seed_engine.cpp
```

### Project Structure

```
openevolve_aetherstate/
├── openevolve/               # OpenEvolve framework (modified)
│   ├── config.py             # Configuration loading
│   ├── controller.py         # Evolution loop controller
│   ├── evaluator.py          # Evaluation orchestration
│   ├── llm/
│   │   └── openai.py         # OpenAI-compatible LLM client
│   ├── prompt/
│   │   ├── sampler.py        # Prompt construction
│   │   └── templates.py      # Template rendering
│   └── database/
│       └── program_db.py     # MAP-Elites program database
├── tasks/aetherstate/        # AetherState task files
│   ├── seed_engine.cpp       # C++ engine (evolvable)
│   ├── chess_runtime.hpp     # Chess rules (fixed)
│   ├── train_loop.py         # Training pipeline
│   ├── evaluate.py           # Evaluator
│   ├── config.yaml           # Evolution config
│   └── outputs/              # Evolution outputs
├── openevolve-run.py         # CLI entry point
└── setup.py                  # Package setup
```

---

## Acknowledgments

Built on top of [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) by Algorithmic Superintelligence. The evolutionary framework, MAP-Elites implementation, and LLM integration are from the OpenEvolve project.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
