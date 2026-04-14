from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACK_VARIANT_TO_CONFIG = {
    ("non_dann", "baseline"): "configs/linux_non_dann_accuracy.yaml",
    ("non_dann", "accuracy"): "configs/linux_non_dann_accuracy.yaml",
    ("non_dann", "balanced"): "configs/linux_non_dann_balanced.yaml",
    ("non_dann", "diversity"): "configs/linux_non_dann.yaml",
    ("non_dann", "short"): "configs/linux_non_dann_short_accuracy.yaml",
    ("non_dann", "short_reg"): "configs/linux_non_dann_short_reg_accuracy.yaml",
    ("non_dann", "short_reg_aug"): "configs/linux_non_dann_short_reg_aug_accuracy.yaml",
    ("dann", "accuracy"): "configs/linux_dann_accuracy.yaml",
    ("dann", "balanced"): "configs/linux_dann_balanced.yaml",
    ("dann", "diversity"): "configs/linux_dann.yaml",
}


def ensure_wifi_pose() -> None:
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    executable = Path(sys.executable)
    if env_name == "WiFiPose" or "WiFiPose" in executable.parts:
        return
    raise RuntimeError(
        "The active environment is not WiFiPose. Activate WiFiPose first, then rerun the formal training command."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Linux formal training track.")
    parser.add_argument("--track", choices=["non_dann", "dann"], default="non_dann")
    parser.add_argument(
        "--variant",
        choices=sorted({variant for _, variant in TRACK_VARIANT_TO_CONFIG.keys()}),
        default="accuracy",
    )
    parser.add_argument("--config", type=str, default=None, help="Override the default track config if needed.")
    parser.add_argument("--experiment_name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_wifi_pose()

    if args.config:
        config_path = args.config
    else:
        key = (args.track, args.variant)
        if key not in TRACK_VARIANT_TO_CONFIG:
            supported = sorted(variant for track, variant in TRACK_VARIANT_TO_CONFIG.keys() if track == args.track)
            raise ValueError(f"Unsupported variant '{args.variant}' for track '{args.track}'. Supported: {supported}")
        config_path = TRACK_VARIANT_TO_CONFIG[key]
    command = [sys.executable, str(PROJECT_ROOT / "train.py"), "--config", config_path]
    if args.experiment_name:
        command.extend(["--experiment_name", args.experiment_name])

    print(f"[formal] track={args.track}")
    print(f"[formal] variant={args.variant}")
    print(f"[formal] cwd={PROJECT_ROOT}")
    print(f"[formal] command={' '.join(command)}")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
