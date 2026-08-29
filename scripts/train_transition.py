#!/usr/bin/env python3
"""Train a transition model on rendered multi-view sequences.

    python scripts/train_transition.py --dataset transphy3d --variant sample
    python scripts/train_transition.py --dataset transphy3d --variant test --epochs 80 --max-pairs 2000
    python scripts/train_transition.py --dataset transphy3d --variant sample --device cpu

This is the Gate-7 training stage. Unlike the synthetic residual, the target here
is not identically zero: TransPhy3D renders transparent and reflective scenes
that rigid reprojection cannot explain, so the residual is a real quantity.

Two things this is NOT, stated up front because both are easy to imply:

* **Not hypothesis-conditioned.** The network input is
  ``(inv_depth, u, v, warped_inv_depth, translation, rotation)`` -- there is no
  hypothesis index. Hypothesis conditioning is the project's central claimed
  mechanism and this model does not have it. It learns *a* transition, not a
  hypothesis-conditioned one.
* **Not real-world imagery.** TransPhy3D is Blender/Cycles rendering. It is
  external to this repository's own simulator, which is what makes the residual
  non-degenerate, but it is not photography.

Every run writes an immutable directory under ``experiments/`` with the same
provenance guarantees as any other experiment -- config, seed, git commit,
environment, the dataset's resolved upstream revision, and the metrics.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import _bootstrap  # noqa: F401
from intervene3d.data.external.loaders import get_reader
from intervene3d.models.torch_transition import (
    TorchResidualConfig,
    build_pair_dataset,
    device_report,
    select_device,
    train_torch_residual,
)
from intervene3d.reproducibility.manifest import finalise_run
from intervene3d.reproducibility.run_dir import create_run_directory
from intervene3d.reproducibility.seeds import set_global_seed
from intervene3d.utils.io import dump_json
from intervene3d.utils.logging import get_logger, setup_logging

LOGGER = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="transphy3d", help="registry key of the training source")
    parser.add_argument("--variant", default="sample", help="acquired variant to read")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3, help="hidden layers")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-pairs", type=int, default=400, help="frame pairs to sample")
    parser.add_argument("--max-points", type=int, default=4096, help="pixels per pair")
    parser.add_argument("--stride", type=int, default=5, help="frame gap = the intervention size")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    parser.add_argument("--experiment-name", default="transition_training")
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--verify", action="store_true", help="recompute dataset checksums first")
    parser.add_argument("--dry-run", action="store_true", help="build supervision, report, train nothing")
    args = parser.parse_args(argv)

    setup_logging()
    seed_report = set_global_seed(args.seed)

    # ------------------------------------------------------------ the data
    reader = get_reader(args.dataset, variant=args.variant)
    verification = reader.verify(full=args.verify)
    provenance = reader.provenance()
    if not verification.get("verified", False):
        LOGGER.error("dataset verification failed: %s", verification)
        return 1
    LOGGER.info(
        "training source: %s/%s @ %s (%s)",
        args.dataset, args.variant, provenance.get("revision"), provenance.get("licence"),
    )

    if not hasattr(reader, "iter_pairs"):
        LOGGER.error(
            "%s exposes no frame pairs with relative pose; the transition model needs "
            "(F_t, a) -> F_t+1 supervision. Use a dataset with per-frame extrinsics "
            "(currently: transphy3d).", args.dataset,
        )
        return 2

    device = select_device(args.device)
    LOGGER.info("device: %s", device_report(device))

    t0 = time.time()
    LOGGER.info("building supervision from up to %d frame pairs (stride=%d)...", args.max_pairs, args.stride)
    X, Y, data_report = build_pair_dataset(
        reader.iter_pairs(stride=args.stride, limit=args.max_pairs),
        max_pairs=args.max_pairs, max_points=args.max_points, seed=args.seed,
    )
    # build_pair_dataset emits one group label per surviving ROW. Reconstructing
    # them here from a parallel per-pair list silently mislabels everything after
    # the first skipped pair, and the held-out-sequence split then leaks.
    row_groups = data_report.pop("row_groups")
    row_consistent = data_report.pop("row_consistent")
    assert len(row_groups) == X.shape[0] == row_consistent.size

    LOGGER.info(
        "supervision: %d samples from %d pairs (%d skipped) across %d sequences, mean valid pixel "
        "fraction %.3f, target RMS %.4g 1/m, built in %.1f s",
        data_report["n_samples"], data_report["pairs_used"], data_report["pairs_skipped"],
        data_report["n_groups"], data_report["mean_valid_pixel_fraction"],
        data_report["target_rms_inv_m"], time.time() - t0,
    )
    if data_report["n_groups"] < 2:
        LOGGER.error(
            "only %d sequence(s) of supervision: a held-out-sequence split is impossible, so any "
            "validation number would be measured on frames from the same sequence as training. "
            "Use a variant with more sequences (transphy3d/test has 28) or raise --max-pairs.",
            data_report["n_groups"],
        )
        return 1
    if data_report["target_rms_inv_m"] == 0.0:
        LOGGER.error(
            "the residual target is identically zero, so there is nothing to learn. That means "
            "rigid reprojection already explains this data exactly -- check the depth scale and "
            "the extrinsics convention before trusting any model trained here."
        )
        return 1

    if args.dry_run:
        print("\n--dry-run: supervision built from real data; no model was trained.")
        print(f"  samples : {data_report['n_samples']:,}  ({X.shape[1]} inputs -> {Y.shape[1]} output)")
        print(f"  sequences: {len(set(row_groups)) if row_groups else 'n/a'}")
        print(f"  target RMS: {data_report['target_rms_inv_m']:.6g} 1/m")
        return 0

    # ------------------------------------------------------------ the run
    config = {
        "experiment": {"name": args.experiment_name, "kind": "transition_training", "seed": args.seed},
        "data": {
            "dataset": args.dataset, "variant": args.variant, "stride": args.stride,
            "max_pairs": args.max_pairs, "max_points": args.max_points,
            "provenance": provenance, "verification": verification,
        },
        "model": {
            "kind": "torch_residual_mlp", "hidden_dim": args.hidden_dim, "depth": args.depth,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "device": args.device, "amp": not args.no_amp,
        },
    }
    # Attach the dataset manifest, or the run ships a dataset_manifest.json that
    # actively asserts "NOT RUN -- no dataset attached" while having trained on
    # 4 GB of it. Same for the reproduction command: the default points at
    # run_experiment.py, which cannot run this.
    argv_repro = (
        f"python scripts/train_transition.py --dataset {args.dataset} --variant {args.variant} "
        f"--seed {args.seed} --epochs {args.epochs} --max-pairs {args.max_pairs} "
        f"--max-points {args.max_points} --stride {args.stride}"
    )
    run_dir = create_run_directory(
        config,
        seed=args.seed,
        root=args.root,
        dataset_manifest={
            "dataset": args.dataset,
            "variant": args.variant,
            **provenance,
            "verification": verification,
            "supervision": {k: v for k, v in data_report.items() if k != "row_groups"},
        },
        extra={"seed_report": seed_report, "reproduction_command": argv_repro},
    )
    LOGGER.info("run directory: %s", run_dir.path)

    cfg = TorchResidualConfig(
        hidden_dim=args.hidden_dim, depth=args.depth, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, batch_size=args.batch_size, epochs=args.epochs,
        device=args.device, amp=not args.no_amp, seed=args.seed,
    )
    try:
        result = train_torch_residual(
            X, Y, groups=row_groups, consistent=row_consistent, cfg=cfg,
            val_fraction=args.val_fraction,
        )
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
        finalise_run(
            run_dir.path, status="failed",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            duration_seconds=round(time.time() - t0, 3), registry_root=args.root,
        )
        raise

    metrics = {
        "experiment": args.experiment_name,
        "seed": args.seed,
        "data": {**data_report, "provenance": provenance, "verification": verification},
        "training": result.report,
    }
    dump_json(run_dir.path / "metrics" / "metrics.json", metrics)
    dump_json(run_dir.path / "checkpoints" / "residual_mlp.json", result.state_dict)

    r = result.report
    summary = "\n".join(
        [
            f"# {args.experiment_name}",
            "",
            f"Source        : {args.dataset}/{args.variant} @ `{provenance.get('revision')}`",
            f"Licence       : {provenance.get('licence')}",
            f"Device        : {r['device'].get('name', r['device']['device'])} "
            f"({r['device'].get('compute_capability', 'cpu')})",
            f"Supervision   : {data_report['n_samples']:,} samples from "
            f"{data_report['pairs_used']} frame pairs, stride {args.stride}",
            f"Split         : {r['split_protocol']}",
            "",
            "| quantity | value |",
            "|---|---|",
            f"| final train MSE | {r['final_train_mse']:.6g} |",
            f"| final val MSE | {r['final_val_mse']:.6g} |",
            f"| rigid (H_D) baseline val MSE | {r['rigid_baseline_val_mse']:.6g} |",
            f"| pooled ratio vs rigid (NOT the result) | {r['val_mse_ratio_vs_rigid_POOLED']} |",
            f"| train seconds | {r['train_seconds']} |",
            "",
            "## The number that matters",
            "",
            "| subset | rows | rigid baseline | model | ratio |",
            "|---|---|---|---|---|",
        ]
        + [
            (f"| {k} | {v['n']:,} ({100 * v['fraction']:.1f}%) | {v['rigid_baseline_mse']:.6g} | "
             f"{v['model_mse']:.6g} | **{v['ratio_vs_rigid']}** |")
            for k, v in (r.get("by_subset") or {}).items() if v
        ]
        + [
            "",
            "Occlusion boundaries carry orders of magnitude more residual than optics does,",
            "so the pooled ratio is dominated by them. **`consistent` is the subset that",
            "speaks to the optics claim**; a ratio above 1 there means the model is worse",
            "than the rigid H_D hypothesis exactly where transparency and reflection live,",
            "however good the pooled number looks.",
        ]
    )
    (run_dir.path / "summary.md").write_text(summary + "\n", encoding="utf-8")
    finalise_run(
        run_dir.path, status="success", metrics_file="metrics/metrics.json",
        summary={
            "final_val_mse": r["final_val_mse"],
            "rigid_baseline_val_mse": r["rigid_baseline_val_mse"],
            "val_mse_ratio_vs_rigid_pooled": r["val_mse_ratio_vs_rigid_POOLED"],
            "by_subset": r.get("by_subset"),
            "n_samples": data_report["n_samples"],
            "device": r["device"].get("name", r["device"]["device"]),
        },
        duration_seconds=round(time.time() - t0, 3), registry_root=args.root,
    )

    print("\n" + "=" * 72)
    print(f"TRAINING RUN SUCCEEDED: {run_dir.path}")
    print("=" * 72)
    print(f"device        : {r['device'].get('name', r['device']['device'])}")
    print(f"samples       : {data_report['n_samples']:,} from {data_report['pairs_used']} frame pairs")
    print(f"val MSE       : {r['final_val_mse']:.6g}   (rigid baseline {r['rigid_baseline_val_mse']:.6g})")
    print(f"pooled ratio  : {r['val_mse_ratio_vs_rigid_POOLED']}   <- dominated by occlusion, not the result")
    for k, v in (r.get("by_subset") or {}).items():
        if v:
            print(f"  {k:20s}: ratio {v['ratio_vs_rigid']}  ({v['n']:,} rows, {100 * v['fraction']:.1f}%)")
    print(f"summary       : {run_dir.path / 'summary.md'}")
    print(f"reproduce     : {argv_repro}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
