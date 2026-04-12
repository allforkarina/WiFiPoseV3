from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    parser.add_argument("--config", type=str, default="configs/windows_smoke.yaml")
    parser.add_argument("--experiment_name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_wifi_pose()

    command = [sys.executable, str(PROJECT_ROOT / "train.py"), "--config", args.config]
    if args.experiment_name:
        command.extend(["--experiment_name", args.experiment_name])

    print(f"[smoke] cwd={PROJECT_ROOT}")
    print(f"[smoke] command={' '.join(command)}")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
