from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACK_VARIANT_TO_CONFIG = {
    ("non_dann", "baseline"): "configs/windows_smoke.yaml",
    ("non_dann", "short"): "configs/windows_smoke_short.yaml",
    ("non_dann", "short_reg"): "configs/windows_smoke_short_reg.yaml",
    ("non_dann", "short_reg_aug"): "configs/windows_smoke_short_reg_aug.yaml",
    ("non_dann", "short_aug"): "configs/windows_smoke_short_aug.yaml",
    ("non_dann", "short_aug_mid"): "configs/windows_smoke_short_aug_mid.yaml",
    ("non_dann", "short_aug_strong"): "configs/windows_smoke_short_aug_strong.yaml",
    ("non_dann", "short_reg_aug_repro"): "configs/windows_smoke_short_reg_aug_repro.yaml",
    ("dann", "baseline"): "configs/windows_smoke_dann.yaml",
}


def ensure_wifi_pose() -> None:
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    executable = Path(sys.executable)
    if env_name == "WiFiPose" or "WiFiPose" in executable.parts:
        return
    raise RuntimeError(
        "The active environment is not WiFiPose. Activate WiFiPose first, then rerun this smoke command."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed Windows smoke training path.")
    parser.add_argument("--track", choices=sorted({track for track, _ in TRACK_VARIANT_TO_CONFIG.keys()}), default="non_dann")
    parser.add_argument(
        "--variant",
        choices=sorted({variant for _, variant in TRACK_VARIANT_TO_CONFIG.keys()}),
        default="baseline",
    )
    parser.add_argument("--config", type=str, default=None)
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

    print(f"[smoke] track={args.track}")
    print(f"[smoke] variant={args.variant}")
    print(f"[smoke] cwd={PROJECT_ROOT}")
    print(f"[smoke] command={' '.join(command)}")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
