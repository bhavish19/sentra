"""
Hyperparameter sweep runner for SENTRA batched MNIST training.

Runs multiple start_all_nodes.py commands with different parameter combinations.
Each run appends metrics into the Excel file via --record-results-xlsx.
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


def _parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_csv_strs(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _build_cmd(args, combo, base_port: int) -> List[str]:
    lr, grad_clip, logit_clip, scale, temp, batch_size, epochs, samples, grad_mode, exp_approx, seed = combo

    cmd = [
        args.python_exe,
        "start_all_nodes.py",
        "--n-nodes",
        str(args.n_nodes),
        "--base-port",
        str(base_port),
        "--batched",
        "--num-epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--mnist-samples",
        str(samples),
        "--learning-rate",
        str(lr),
        "--loss-mode",
        args.loss_mode,
        "--field-size",
        str(args.field_size),
        "--scale-factor",
        str(scale),
        "--softmax-temperature",
        str(temp),
        "--exp-approx",
        exp_approx,
        "--softmax-grad-mode",
        grad_mode,
        "--grad-clip",
        str(grad_clip),
        "--logit-clip",
        str(logit_clip),
        "--explode-logit-threshold",
        str(args.explode_logit_threshold),
        "--loss-growth-threshold",
        str(args.loss_growth_threshold),
        "--grad-norm-threshold",
        str(args.grad_norm_threshold),
        "--seed",
        str(seed),
        "--record-results-xlsx",
        args.record_results_xlsx,
    ]

    if args.no_abort_on_instability:
        cmd.append("--no-abort-on-instability")
    if args.debug_numerics:
        cmd.append("--debug-numerics")
    if args.debug_division:
        cmd.append("--debug-division")
    if args.export_reconstructed_model:
        model_path = Path(args.export_reconstructed_model)
        run_suffix = f"_p{base_port}_s{seed}"
        run_model_path = model_path.with_name(f"{model_path.stem}{run_suffix}{model_path.suffix}")
        cmd.extend(["--export-reconstructed-model", str(run_model_path)])
        cmd.extend(["--export-timeout", str(args.export_timeout)])

    return cmd


def _iter_combos(args) -> Iterable[tuple]:
    return itertools.product(
        args.learning_rates,
        args.grad_clips,
        args.logit_clips,
        args.scale_factors,
        args.softmax_temperatures,
        args.batch_sizes,
        args.epochs_list,
        args.mnist_samples_list,
        args.softmax_grad_modes,
        args.exp_approxs,
        args.seeds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SENTRA hyperparameter sweep and append results to Excel.")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable to use (default: current interpreter)")
    parser.add_argument("--n-nodes", type=int, default=3)
    parser.add_argument("--field-size", type=int, default=2305843009213693951)
    parser.add_argument("--loss-mode", choices=["mse", "softmax"], default="softmax")
    parser.add_argument("--explode-logit-threshold", type=float, default=100.0)
    parser.add_argument("--loss-growth-threshold", type=float, default=5.0)
    parser.add_argument("--grad-norm-threshold", type=float, default=1500.0)
    parser.add_argument("--no-abort-on-instability", action="store_true", default=True)
    parser.add_argument("--debug-numerics", action="store_true")
    parser.add_argument("--debug-division", action="store_true")
    parser.add_argument("--record-results-xlsx", default="logs/sentra_runs.xlsx")
    parser.add_argument("--export-reconstructed-model", default="", help="Optional base path for per-run reconstructed model export (.npz)")
    parser.add_argument("--export-timeout", type=float, default=180.0)

    parser.add_argument("--base-port-start", type=int, default=12000)
    parser.add_argument("--base-port-step", type=int, default=20)
    parser.add_argument("--sleep-between-runs", type=float, default=2.0)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means no cap")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")

    parser.add_argument("--learning-rates", type=_parse_csv_floats, default=[0.003], help="CSV, e.g. 0.001,0.002,0.003")
    parser.add_argument("--grad-clips", type=_parse_csv_floats, default=[0.5], help="CSV")
    parser.add_argument("--logit-clips", type=_parse_csv_floats, default=[4.0], help="CSV")
    parser.add_argument("--scale-factors", type=_parse_csv_ints, default=[65536], help="CSV")
    parser.add_argument("--softmax-temperatures", type=_parse_csv_ints, default=[2], help="CSV of ints (fixed-point path)")
    parser.add_argument("--batch-sizes", type=_parse_csv_ints, default=[64], help="CSV")
    parser.add_argument("--epochs-list", type=_parse_csv_ints, default=[8], help="CSV")
    parser.add_argument("--mnist-samples-list", type=_parse_csv_ints, default=[10000], help="CSV")
    parser.add_argument("--softmax-grad-modes", type=_parse_csv_strs, default=["secure_approx"], help="CSV: secure_approx,opened_exact")
    parser.add_argument("--exp-approxs", type=_parse_csv_strs, default=["pade22"], help="CSV: pade22,taylor5")
    parser.add_argument("--seeds", type=_parse_csv_ints, default=[2026], help="CSV")

    args = parser.parse_args()

    combos = list(_iter_combos(args))
    if args.max_runs and args.max_runs > 0:
        combos = combos[: args.max_runs]

    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path("logs") / f"sweep_{sweep_id}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "num_combos": len(combos),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Sweep ID: {sweep_id}")
    print(f"Total runs: {len(combos)}")
    print(f"Excel: {args.record_results_xlsx}")
    print(f"Sweep config: {sweep_dir / 'config.json'}")

    failures = 0
    for i, combo in enumerate(combos, start=1):
        base_port = args.base_port_start + (i - 1) * args.base_port_step
        cmd = _build_cmd(args, combo, base_port)
        cmd_str = " ".join(cmd)
        print("\n" + "=" * 72)
        print(f"Run {i}/{len(combos)} | base_port={base_port}")
        print(cmd_str)
        print("=" * 72)

        if args.dry_run:
            continue

        rc = subprocess.call(cmd)
        print(f"Run {i} exit code: {rc}")
        if rc != 0:
            failures += 1
            if not args.continue_on_error:
                print("Stopping sweep due to failure. Use --continue-on-error to keep going.")
                break

        if i < len(combos):
            time.sleep(max(0.0, args.sleep_between_runs))

    summary = {
        "sweep_id": sweep_id,
        "requested_runs": len(combos),
        "failures": failures,
        "status": "ok" if failures == 0 else "completed_with_failures",
    }
    (sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSweep finished.")
    print(json.dumps(summary, indent=2))
    print(f"Summary: {sweep_dir / 'summary.json'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
