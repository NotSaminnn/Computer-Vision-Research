#!/usr/bin/env python3
"""Generate the synthetic Intervene3D benchmark.

    python scripts/generate_synthetic_data.py --config configs/synthetic/smoke.yaml
    python scripts/generate_synthetic_data.py --config configs/synthetic/phase1.yaml --seed 7
    python scripts/generate_synthetic_data.py --config configs/synthetic/smoke.yaml --dry-run

The generator is deterministic under ``dataset.seed``: the same config and seed
always produce byte-identical arrays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from intervene3d.config.loader import config_hash, load_config
from intervene3d.config.schema import validate_synthetic_config
from intervene3d.data.synthetic.dataset_writer import generate_dataset
from intervene3d.data.synthetic.validator import validate_dataset
from intervene3d.reproducibility.seeds import set_global_seed
from intervene3d.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="synthetic generator YAML config")
    parser.add_argument("--seed", type=int, default=None, help="override dataset.seed")
    parser.add_argument("--output-root", default=None, help="override output.root")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="dotted.key=value override")
    parser.add_argument("--dry-run", action="store_true", help="validate and report; write nothing")
    parser.add_argument("--force", action="store_true", help="regenerate even if the dataset already exists")
    parser.add_argument("--no-validate", action="store_true", help="skip post-generation validation")
    args = parser.parse_args(argv)

    setup_logging()
    config = validate_synthetic_config(load_config(args.config, overrides=args.overrides))
    if args.seed is not None:
        config["dataset"]["seed"] = int(args.seed)
    root = Path(args.output_root or config["output"]["root"]) / config["dataset"]["name"]

    n_variants = int(config["dataset"]["num_base_scenes"]) * len(config["mechanisms"])
    print(f"dataset        : {config['dataset']['name']} (v{config['dataset'].get('version', '?')})")
    print(f"config hash    : {config_hash(config)}")
    print(f"seed           : {config['dataset']['seed']}")
    print(f"base scenes    : {config['dataset']['num_base_scenes']}")
    print(f"mechanisms     : {config['mechanisms']}")
    print(f"scene variants : {n_variants}")
    print(f"landmarks/scene: {config['scene']['num_content_landmarks']} content + 4 frame + "
          f"{config['observer_markers']['count']} markers")
    print(f"output         : {root}")

    if args.dry_run:
        print("\n--dry-run: configuration is valid; nothing was written.")
        return 0

    if root.exists() and not args.force:
        if (root / "manifest.json").exists():
            print(f"\ndataset already exists at {root} -- pass --force to regenerate.")
            if not args.no_validate:
                report = validate_dataset(root)
                print()
                print(report.render())
                return 0 if report.passed else 1
            return 0

    set_global_seed(int(config["dataset"]["seed"]))
    manifest = generate_dataset(config, output_root=args.output_root)
    stats = manifest["statistics"]
    print(
        f"\ngenerated {stats['n_scene_variants']} scene variants over {stats['n_base_scenes']} base scenes\n"
        f"  resolvable      : {stats['resolvable_count']}\n"
        f"  non-identifiable: {stats['non_resolvable_count']} "
        f"({100 * (1 - stats['resolvable_fraction']):.1f}%)\n"
        f"  median MCRB     : {stats['mcrb_median']:.4f} m"
    )

    if args.no_validate:
        return 0
    report = validate_dataset(root)
    print()
    print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
