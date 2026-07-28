"""Unit tests for the AetherState trainability evaluator.

These tests target the pure logic in ``tasks/aetherstate/evaluate.py``:

  - bundle parsing
  - architecture validation
  - multi-seed metric aggregation
  - Tier-2 gating / metric replacement

Run with::

    python -m unittest tasks.aetherstate.test_evaluate

from the project root.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure the project root is on sys.path so evaluate.py can import openevolve.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_EVAL_PATH = Path(__file__).resolve().parent / "evaluate.py"
_eval_spec = importlib.util.spec_from_file_location("aetherstate_evaluate", _EVAL_PATH)
_evaluate = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_evaluate)


def _valid_bundle_text() -> str:
    return r"""ARCHITECTURE = {
    "input_features": 768,
    "accumulator_size": 256,
    "hidden_size": 32,
    "output_slots": 4096,
    "move_stride": 64,
    "quant_shift": 7,
    "weight_magic": "AESTATEW",
    "max_features_per_record": 32,
    "weight_version": 1,
}
SEED_ENGINE_CPP = r'''int main() { return 0; }'''
TRAIN_LOOP_PY = r'''# pass'''
RESEARCH_NOTEBOOK = r'''## notebook'''
"""


class TestBundleParsing(unittest.TestCase):
    def test_valid_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.py"
            path.write_text(_valid_bundle_text(), encoding="utf-8")
            bundle = _evaluate._parse_bundle(str(path))
            self.assertEqual(bundle["architecture"]["hidden_size"], 32)
            self.assertIn("seed_engine_cpp", bundle)
            self.assertIn("train_loop_py", bundle)
            self.assertIn("research_notebook", bundle)

    def test_missing_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.py"
            with self.assertRaises(ValueError) as ctx:
                _evaluate._parse_bundle(str(path))
            self.assertIn("Bundle not found", str(ctx.exception))

    def test_invalid_python_syntax(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.py"
            path.write_text("this is not valid python", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                _evaluate._parse_bundle(str(path))
            self.assertIn("syntax error", str(ctx.exception).lower())

    def test_missing_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.py"
            path.write_text(
                'SEED_ENGINE_CPP = r\'\'\'int main() { return 0; }\'\'\'\n'
                'TRAIN_LOOP_PY = r\'\'\'# pass\'\'\'\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                _evaluate._parse_bundle(str(path))
            self.assertIn("ARCHITECTURE", str(ctx.exception))


class TestArchitectureValidation(unittest.TestCase):
    def _valid_architecture(self):
        return {
            "input_features": 768,
            "accumulator_size": 256,
            "hidden_size": 32,
            "output_slots": 4096,
            "move_stride": 64,
            "quant_shift": 7,
            "weight_magic": "AESTATEW",
            "max_features_per_record": 32,
            "weight_version": 1,
        }

    def test_valid_architecture(self):
        ok, err = _evaluate._validate_architecture(self._valid_architecture())
        self.assertTrue(ok, err)
        self.assertEqual(err, "")

    def test_missing_key(self):
        arch = self._valid_architecture()
        del arch["hidden_size"]
        ok, err = _evaluate._validate_architecture(arch)
        self.assertFalse(ok)
        self.assertIn("hidden_size", err)

    def test_hidden_size_out_of_bounds(self):
        arch = self._valid_architecture()
        arch["hidden_size"] = 4096
        ok, err = _evaluate._validate_architecture(arch)
        self.assertFalse(ok)
        self.assertIn("hidden_size", err)

    def test_wrong_weight_magic(self):
        arch = self._valid_architecture()
        arch["weight_magic"] = "HACKED"
        ok, err = _evaluate._validate_architecture(arch)
        self.assertFalse(ok)
        self.assertIn("AESTATEW", err)

    def test_output_slots_mismatch(self):
        arch = self._valid_architecture()
        arch["move_stride"] = 32
        ok, err = _evaluate._validate_architecture(arch)
        self.assertFalse(ok)
        self.assertIn("move_stride", err)


class TestMultiSeedAggregation(unittest.TestCase):
    def test_aggregate_single_seed(self):
        results = [{"win_rate": 0.6, "nodes_per_second": 1e6}]
        metrics = _evaluate._aggregate_results(results)
        self.assertAlmostEqual(metrics["mean_win_rate"], 0.6)
        self.assertAlmostEqual(metrics["win_rate"], 0.6)

    def test_aggregate_multiple_seeds(self):
        results = [
            {"win_rate": 0.5, "nodes_per_second": 1e6, "draw_rate": 0.1, "avg_game_length": 60, "val_loss": 0.3},
            {"win_rate": 0.7, "nodes_per_second": 2e6, "draw_rate": 0.2, "avg_game_length": 50, "val_loss": 0.2},
            {"win_rate": 0.6, "nodes_per_second": 1.5e6, "draw_rate": 0.15, "avg_game_length": 55, "val_loss": 0.25},
        ]
        metrics = _evaluate._aggregate_results(results)
        self.assertAlmostEqual(metrics["mean_win_rate"], 0.6)
        self.assertGreater(metrics["std_win_rate"], 0.0)
        self.assertAlmostEqual(metrics["win_rate"], 0.6)
        self.assertAlmostEqual(metrics["mean_nodes_per_second"], 1.5e6)
        self.assertAlmostEqual(metrics["nodes_per_second"], 1.5e6)

    def test_aggregate_missing_keys_defaults_to_zero(self):
        results = [{}]
        metrics = _evaluate._aggregate_results(results)
        self.assertEqual(metrics["mean_win_rate"], 0.0)
        self.assertEqual(metrics["win_rate"], 0.0)


class TestTier2Gating(unittest.TestCase):
    def test_gate_passes_when_enabled_and_win_rate_above_threshold(self):
        self.assertTrue(_evaluate._tier2_gate_passes(0.75, True, 0.6))

    def test_gate_fails_when_disabled(self):
        self.assertFalse(_evaluate._tier2_gate_passes(0.75, False, 0.6))

    def test_gate_fails_when_win_rate_below_threshold(self):
        self.assertFalse(_evaluate._tier2_gate_passes(0.55, True, 0.6))

    def test_apply_tier2_keeps_tier1_when_not_replacing(self):
        metrics = {"win_rate": 0.6}
        t2 = {"win_rate": 0.9, "nodes_per_second": 2e6, "draw_rate": 0.1, "avg_game_length": 55, "val_loss": 0.1}
        updated = _evaluate._apply_tier2_metrics(metrics, t2, tier2_replaces=False)
        self.assertAlmostEqual(updated["tier2_win_rate"], 0.9)
        self.assertAlmostEqual(updated["win_rate"], 0.6)

    def test_apply_tier2_replaces_tier1_when_requested(self):
        metrics = {"win_rate": 0.6}
        t2 = {"win_rate": 0.9, "nodes_per_second": 2e6, "draw_rate": 0.1, "avg_game_length": 55, "val_loss": 0.1}
        updated = _evaluate._apply_tier2_metrics(metrics, t2, tier2_replaces=True)
        self.assertAlmostEqual(updated["win_rate"], 0.9)
        self.assertAlmostEqual(updated["nodes_per_second"], 2e6)


if __name__ == "__main__":
    unittest.main()
