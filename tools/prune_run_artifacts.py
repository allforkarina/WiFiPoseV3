from __future__ import annotations

import argparse
import os
import stat
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def train_run_key(path: Path) -> str:
    stem = path.stem
    for suffix in ("_history", "_curves"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, onerror=handle_remove_readonly)
    elif path.exists():
        path.unlink()


def handle_remove_readonly(func, path, exc_info) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def prune_train_logs(train_dir: Path, keep: int) -> None:
    if not train_dir.exists():
        return
    grouped: dict[str, list[Path]] = {}
    for path in train_dir.iterdir():
        if not path.is_file():
            continue
        grouped.setdefault(train_run_key(path), []).append(path)
    ordered = sorted(
        grouped.items(),
        key=lambda item: max(p.stat().st_mtime for p in item[1]),
        reverse=True,
    )
    for _, paths in ordered[keep:]:
        for path in paths:
            delete_path(path)


def prune_checkpoint_groups(checkpoint_dir: Path, keep: int) -> None:
    if not checkpoint_dir.exists():
        return
    grouped: dict[str, list[Path]] = {}
    for path in checkpoint_dir.glob("*.pth"):
        stem = path.stem[:-5] if path.stem.endswith("_last") else path.stem
        grouped.setdefault(stem, []).append(path)
    ordered = sorted(
        grouped.items(),
        key=lambda item: max(p.stat().st_mtime for p in item[1]),
        reverse=True,
    )
    for _, paths in ordered[keep:]:
        for path in paths:
            delete_path(path)


def prune_subdirs(parent: Path, keep: int, prefix: str | None = None) -> None:
    if not parent.exists():
        return
    paths = [p for p in parent.iterdir() if p.is_dir() and (prefix is None or p.name.startswith(prefix))]
    ordered = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    for path in ordered[keep:]:
        delete_path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep only the most recent N training, eval, diagnose, and checkpoint artifacts.")
    parser.add_argument("--keep", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep = max(1, int(args.keep))

    prune_train_logs(PROJECT_ROOT / "logs" / "train", keep)
    prune_subdirs(PROJECT_ROOT / "logs" / "eval", keep)
    prune_checkpoint_groups(PROJECT_ROOT / "checkpoints", keep)
    print(f"[prune] keep={keep} completed")


if __name__ == "__main__":
    main()
