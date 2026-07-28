#!/usr/bin/env python3
"""
AetherState Bundle Unpacker
---------------------------
Extracts the human-readable source files from an ``aetherstate_bundle.py``
file for inspection or manual editing.

Usage:
    python unpack_bundle.py [path/to/aetherstate_bundle.py]

Defaults to ``tasks/aetherstate/aetherstate_bundle.py`` relative to this
script.  Writes ``seed_engine.cpp`` and ``train_loop.py`` next to the bundle.
"""

import ast
import argparse
import sys
from pathlib import Path


def unpack_bundle(bundle_path: Path) -> tuple[dict, str, str]:
    """Parse the bundle and return (architecture, seed_engine_cpp, train_loop_py)."""
    source = bundle_path.read_text(encoding="utf-8")

    # Try AST literal eval first for safety, fall back to exec for raw strings.
    tree = ast.parse(source)
    namespace: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        namespace[target.id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass

    if "ARCHITECTURE" not in namespace or "SEED_ENGINE_CPP" not in namespace or "TRAIN_LOOP_PY" not in namespace:
        exec(compile(source, bundle_path, "exec"), namespace)  # noqa: S102

    return namespace["ARCHITECTURE"], namespace["SEED_ENGINE_CPP"], namespace["TRAIN_LOOP_PY"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unpack AetherState bundle into source files")
    parser.add_argument(
        "bundle_path",
        type=Path,
        nargs="?",
        default=None,
        help="Path to aetherstate_bundle.py (default: tasks/aetherstate/aetherstate_bundle.py)",
    )
    args = parser.parse_args(argv)

    if args.bundle_path is None:
        # Default to the bundle next to this script.
        args.bundle_path = Path(__file__).resolve().parent / "aetherstate_bundle.py"

    if not args.bundle_path.exists():
        print(f"Bundle not found: {args.bundle_path}", file=sys.stderr)
        return 1

    architecture, seed_engine_cpp, train_loop_py = unpack_bundle(args.bundle_path)
    out_dir = args.bundle_path.parent

    (out_dir / "seed_engine.cpp").write_text(seed_engine_cpp, encoding="utf-8")
    (out_dir / "train_loop.py").write_text(train_loop_py, encoding="utf-8")

    print(f"Unpacked {args.bundle_path} to:")
    print(f"  {out_dir / 'seed_engine.cpp'}")
    print(f"  {out_dir / 'train_loop.py'}")
    print("Architecture:")
    for key, value in architecture.items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
