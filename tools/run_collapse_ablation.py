from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_variants() -> dict[str, list[str]]:
    return {
        "pelvis_base": [
            "--normalize_mode", "pelvis_torso",
            "--lambda_var", "0.0",
            "--lambda_batch_div", "0.0",
            "--lambda_inter_div", "0.0",
            "--lambda_intra_div", "0.0",
            "--selection_mode", "accuracy",
            "--disable_batch_div_schedule",
            "--disable_inter_div_schedule",
            "--disable_intra_div_schedule",
        ],
        "pelvis_interdiv": [
            "--normalize_mode", "pelvis_torso",
            "--lambda_var", "0.0",
            "--lambda_batch_div", "0.0",
            "--lambda_inter_div", "0.25",
            "--lambda_intra_div", "0.0",
            "--selection_mode", "accuracy",
            "--disable_batch_div_schedule",
            "--disable_inter_div_schedule",
            "--disable_intra_div_schedule",
        ],
        "pelvis_interdiv_divsel": [
            "--normalize_mode", "pelvis_torso",
            "--lambda_var", "0.0",
            "--lambda_batch_div", "0.0",
            "--lambda_inter_div", "0.25",
            "--lambda_intra_div", "0.0",
            "--selection_mode", "diversity_first",
            "--disable_batch_div_schedule",
            "--disable_inter_div_schedule",
            "--disable_intra_div_schedule",
        ],
        "meanrms_base": [
            "--normalize_mode", "mean_rms",
            "--lambda_var", "0.0",
            "--lambda_batch_div", "0.0",
            "--lambda_inter_div", "0.0",
            "--lambda_intra_div", "0.0",
            "--selection_mode", "accuracy",
            "--disable_batch_div_schedule",
            "--disable_inter_div_schedule",
            "--disable_intra_div_schedule",
        ],
        "meanrms_interdiv": [
            "--normalize_mode", "mean_rms",
            "--lambda_var", "0.0",
            "--lambda_batch_div", "0.0",
            "--lambda_inter_div", "0.25",
            "--lambda_intra_div", "0.0",
            "--selection_mode", "accuracy",
            "--disable_batch_div_schedule",
            "--disable_inter_div_schedule",
            "--disable_intra_div_schedule",
        ],
        "meanrms_interdiv_divsel": [
            "--normalize_mode", "mean_rms",
            "--lambda_var", "0.0",
            "--lambda_batch_div", "0.0",
            "--lambda_inter_div", "0.25",
            "--lambda_intra_div", "0.0",
            "--selection_mode", "diversity_first",
            "--disable_batch_div_schedule",
            "--disable_inter_div_schedule",
            "--disable_intra_div_schedule",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal-variable ablations for average-pose collapse diagnosis.")
    parser.add_argument("--variants", nargs="*", default=None, help="Subset of variants to run. Default: all.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model_name", default="resnet1d")
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--device", default=None, help="Reserved for future use; train.py reads device from config.")
    parser.add_argument("--aoa_cache_root", default=None)
    parser.add_argument("--labels_root", default=None)
    parser.add_argument("--val_env", default="env3")
    parser.add_argument("--test_env", default="env4")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = build_variants()
    selected = args.variants or list(variants.keys())

    unknown = [name for name in selected if name not in variants]
    if unknown:
        raise SystemExit(f"Unknown variants: {', '.join(unknown)}")

    for name in selected:
        checkpoint_path = PROJECT_ROOT / "checkpoints" / f"ablation_{name}.pth"
        cmd = [
            args.python,
            "train.py",
            "--config", args.config,
            "--model_name", args.model_name,
            "--window_size", str(args.window_size),
            "--epochs", str(args.epochs),
            "--max_steps", str(args.max_steps),
            "--checkpoint", str(checkpoint_path),
            "--val_env", args.val_env,
            "--test_env", args.test_env,
            "--experiment_name", f"ablation_{name}",
        ]
        if args.aoa_cache_root:
            cmd.extend(["--aoa_cache_root", args.aoa_cache_root])
        if args.labels_root:
            cmd.extend(["--labels_root", args.labels_root])
        cmd.extend(variants[name])

        print(f"[ablation] variant={name}")
        print("[ablation] command=" + " ".join(cmd))
        if args.dry_run:
            continue

        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
