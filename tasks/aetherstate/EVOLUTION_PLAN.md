# AetherState Co-Evolution Implementation Plan

## Goal
Transform the AetherState task so OpenEvolve can co-evolve the C++ inference engine and the Python training loop, keep architecture dimensions in sync, and evaluate mutations on structural/trainability criteria without real training or chess games during evolution.

## Design Decisions

### 1. Evolution evaluator scope
- **Remove `random_time` and `selfplay_time` win-rate games from the evolution evaluator.**
- The evolution evaluator will only check: compiles, generates data, micro-trains, exports/loads weights, and inference speed.
- A separate **post-evolution validation script** (`final_validate.py`) will run real training and real chess games on the best program after evolution is done. This is reserved for the expensive final run.

### 2. Source of truth
- `aetherstate_bundle.py` is the OpenEvolve initial program and the source of truth.
- `seed_engine.cpp` and `train_loop.py` become generated artifacts for human readability/debugging.
- A helper script, `unpack_bundle.py`, can extract the bundle into the two source files.

## Step-by-Step Plan

### Step 1: Create `aetherstate_bundle.py`
- Single Python file containing:
  - `ARCHITECTURE` dict (input_features, accumulator_size, hidden_size, output_slots, quant_shift, weight_magic, etc.)
  - `SEED_ENGINE_CPP` raw string with the current `seed_engine.cpp` content, using placeholders for architecture constants
  - `TRAIN_LOOP_PY` raw string with the current `train_loop.py` content, using placeholders for architecture constants
- The bundle must be valid Python and parseable by `ast.literal_eval` / `exec`.

### Step 2: Refactor `seed_engine.cpp`
- Replace hardcoded constants with values injected by the evaluator.
- Keep the `// # EVOLVE-BLOCK-START` ... `// # EVOLVE-BLOCK-END` markers around the mutable neural-network and data-generation code.
- Ensure the weight file magic and `NeuralNet` struct layout remain under guardrail control.

### Step 3: Refactor `train_loop.py`
- Replace hardcoded constants with values read from an injected `arch_constants.py` or environment variables.
- Keep the PyTorch model, training loop, and weight export/import logic parametric.
- Add a `--no-benchmark` fast path for trainability-only evaluation.

### Step 4: Rewrite `tasks/aetherstate/evaluate.py`
Implement the following pipeline, failing fast and returning zeroed metrics on any failure:

1. **Parse the bundle** – extract `ARCHITECTURE`, `SEED_ENGINE_CPP`, `TRAIN_LOOP_PY`.
2. **Validate architecture** – required keys present, sizes within bounds.
3. **Write files** into a temp dir:
   - `seed_engine.cpp` with architecture constants prepended/injected
   - `train_loop.py` with architecture constants injected
   - `chess_runtime.hpp` copied from task dir
4. **Compile C++ engine**.
5. **Python syntax check** on `train_loop.py`.
6. **Generate data** – run `aetherstate generate_data 10`, verify non-empty output.
7. **Micro-train** – run `train_loop.py` with `--selfplay-games 1 --epochs 1 --batch-size 2 --no-benchmark`, verify no crash and no NaN.
8. **Export/load weights** – verify `weights.bin` produced and C++ engine loads it.
9. **Inference speed** – run `bench_time 1` with loaded weights, record `nodes_per_second`.

Return metrics:
```python
{
    "fitness": nodes_per_second / 1_000_000 * trainability_ok,
    "combined_score": <same as fitness>,
    "compile_ok": 1.0 / 0.0,
    "generate_data_ok": 1.0 / 0.0,
    "train_start_ok": 1.0 / 0.0,
    "weights_export_ok": 1.0 / 0.0,
    "weights_load_ok": 1.0 / 0.0,
    "nodes_per_second": <nps>,
    "trainable": 1.0 / 0.0,
}
```

### Step 5: Update `tasks/aetherstate/config.yaml`
- Replace the C++/SIMD-only system message with one that authorizes the LLM to mutate:
  - The `ARCHITECTURE` dict.
  - The C++ inference engine (`SEED_ENGINE_CPP`).
  - The Python training algorithm (`TRAIN_LOOP_PY`).
- Define a file-targeted SEARCH/REPLACE convention.
- Add explicit guardrail reminders (do not remove delimiters, do not change weight magic, do not explode sizes).

### Step 6: Guardrails in `evaluate.py`
- **Architecture bounds**: `accumulator_size <= 1024`, `hidden_size <= 256`, `output_slots == 4096`, `quant_shift` in safe range.
- **Weight format protection**: require `weight_magic == "AESTATEW"`, reject changes to the training-record struct or weight-export layout.
- **Timeout enforcement**: strict per-step timeouts.
- **Sandboxing**: run `train_loop.py` via `subprocess`, never import it.
- **Evaluator escape prevention**: reject writes outside the temp dir, network access, or unexpected imports.

### Step 7: Create `unpack_bundle.py`
- Reads `aetherstate_bundle.py` and writes `seed_engine.cpp` and `train_loop.py` for inspection.

### Step 8: Validation
- Run `python tasks/aetherstate/evaluate.py tasks/aetherstate/aetherstate_bundle.py` end-to-end.
- Run `python -m unittest discover tests` or targeted smoke tests to ensure OpenEvolve core is unaffected.
- Run `python tasks/aetherstate/unpack_bundle.py` and verify the generated files compile manually.

### Step 9: Code Review
- Spawn a code reviewer over the changed files.
- Fix any issues before declaring complete.

## Files Changed
- `tasks/aetherstate/seed_engine.cpp`
- `tasks/aetherstate/train_loop.py`
- `tasks/aetherstate/evaluate.py`
- `tasks/aetherstate/config.yaml`
- `tasks/aetherstate/aetherstate_bundle.py` (new)
- `tasks/aetherstate/unpack_bundle.py` (new)
- `tasks/aetherstate/EVOLUTION_PLAN.md` (this file)
