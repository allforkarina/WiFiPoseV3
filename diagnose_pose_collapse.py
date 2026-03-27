from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataloader.aoa_dataset import AOASampleDataset, sample_to_env
from train import PROJECT_ROOT, build_model, load_config, resolve_data_roots, resolve_normalize_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose average-pose collapse across checkpoints.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", action="append", default=[], help="Checkpoint path; can be passed multiple times.")
    parser.add_argument("--checkpoint_glob", type=str, default="checkpoints/*.pth")
    parser.add_argument("--aoa_cache_root", type=str, default=None)
    parser.add_argument("--labels_root", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--normalize_mode", type=str, choices=["pelvis_torso", "mean_rms"], default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sample_stride", type=int, default=200)
    parser.add_argument("--max_samples", type=int, default=1600)
    parser.add_argument("--val_env", type=str, default=None)
    parser.add_argument("--test_env", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="logs/diagnose_pose_collapse")
    return parser.parse_args()


def resolve_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for item in args.checkpoint:
        path = Path(item)
        paths.append(path if path.is_absolute() else PROJECT_ROOT / path)
    if not paths:
        glob_root = Path(args.checkpoint_glob)
        if not glob_root.is_absolute():
            glob_root = PROJECT_ROOT / args.checkpoint_glob
        paths = sorted(PROJECT_ROOT.glob(args.checkpoint_glob))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        unique.append(path)
    if not unique:
        raise FileNotFoundError("No checkpoints found for diagnosis.")
    return unique


def choose_indices(ds: AOASampleDataset, sample_stride: int, max_samples: int, val_env: str | None, test_env: str | None) -> list[int]:
    chosen: list[int] = []
    for idx, (_, sample, _) in enumerate(ds.index):
        env_id = sample_to_env(sample)
        if val_env is not None and env_id != val_env:
            continue
        if test_env is not None and env_id != test_env:
            continue
        if idx % max(1, sample_stride) != 0:
            continue
        chosen.append(idx)
        if len(chosen) >= max_samples:
            break
    if not chosen:
        raise RuntimeError("No samples selected; adjust env filters or sample stride.")
    return chosen


def summarize_checkpoint(
    checkpoint_path: Path,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    ds: AOASampleDataset,
    indices: list[int],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_cfg = ckpt.get("config", cfg)
    window_size = int(
        args.window_size
        if args.window_size is not None
        else ckpt_cfg.get("dataset", {}).get("window_size", cfg.get("dataset", {}).get("window_size", 1))
    )
    model = build_model(ckpt_cfg, device, args.model_name, window_size=window_size)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    targets: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    actions: list[str] = []
    with torch.no_grad():
        for idx in indices:
            x, y, meta = ds[idx]
            pred = model(x.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            preds.append(pred)
            targets.append(y.numpy())
            actions.append(str(meta["action"]))

    target_arr = np.stack(targets)
    pred_arr = np.stack(preds)
    action_rows: list[dict[str, Any]] = []
    target_mean_pose = target_arr.mean(axis=0)
    pred_mean_pose = pred_arr.mean(axis=0)
    target_std_flat = target_arr.reshape(len(indices), -1).std(axis=0)
    pred_std_flat = pred_arr.reshape(len(indices), -1).std(axis=0)

    grouped_target: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_pred: dict[str, list[np.ndarray]] = defaultdict(list)
    for action, target_pose, pred_pose in zip(actions, target_arr, pred_arr):
        grouped_target[action].append(target_pose)
        grouped_pred[action].append(pred_pose)

    for action in sorted(grouped_target):
        action_target = np.stack(grouped_target[action])
        action_pred = np.stack(grouped_pred[action])
        action_rows.append(
            {
                "checkpoint": checkpoint_path.name,
                "action": action,
                "samples": int(action_target.shape[0]),
                "target_group_std_mean": float(action_target.reshape(action_target.shape[0], -1).std(axis=0).mean()),
                "pred_group_std_mean": float(action_pred.reshape(action_pred.shape[0], -1).std(axis=0).mean()),
                "mean_pose_gap": float(np.sqrt(((action_pred.mean(axis=0) - action_target.mean(axis=0)) ** 2).sum(axis=1)).mean()),
                "mean_pose_to_global_pred_gap": float(
                    np.sqrt(((action_pred.mean(axis=0) - pred_mean_pose) ** 2).sum(axis=1)).mean()
                ),
            }
        )

    mean_pose_baseline = np.broadcast_to(target_mean_pose[None, ...], target_arr.shape)
    summary = {
        "checkpoint": checkpoint_path.name,
        "samples": len(indices),
        "target_global_mean_abs": float(np.abs(target_mean_pose).mean()),
        "pred_global_mean_abs": float(np.abs(pred_mean_pose).mean()),
        "target_sample_std_mean": float(target_arr.reshape(len(indices), -1).std(axis=1).mean()),
        "pred_sample_std_mean": float(pred_arr.reshape(len(indices), -1).std(axis=1).mean()),
        "target_across_dataset_std_mean": float(target_std_flat.mean()),
        "pred_across_dataset_std_mean": float(pred_std_flat.mean()),
        "variance_ratio_pred_over_target": float(pred_std_flat.mean() / (target_std_flat.mean() + 1e-8)),
        "mse_pred_to_target": float(((target_arr - pred_arr) ** 2).mean()),
        "mse_meanpose_to_target": float(((target_arr - mean_pose_baseline) ** 2).mean()),
        "pred_target_mean_pose_gap": float(np.sqrt(((pred_mean_pose - target_mean_pose) ** 2).sum(axis=1)).mean()),
    }
    return summary, action_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(config_path)
    aoa_root, labels_root = resolve_data_roots(cfg, args.aoa_cache_root, args.labels_root)
    checkpoint_paths = resolve_checkpoint_paths(args)
    device = torch.device(args.device)
    dataset_cache: dict[tuple[int, str], AOASampleDataset] = {}
    default_window_size = int(args.window_size if args.window_size is not None else cfg.get("dataset", {}).get("window_size", 1))
    default_normalize_mode = args.normalize_mode or resolve_normalize_mode(cfg)
    default_ds = AOASampleDataset(
        aoa_root=aoa_root,
        labels_root=labels_root,
        window_size=default_window_size,
        normalize_mode=default_normalize_mode,
    )
    dataset_cache[(default_window_size, default_normalize_mode)] = default_ds
    indices = choose_indices(default_ds, args.sample_stride, args.max_samples, args.val_env, args.test_env)

    summary_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        ckpt_cfg = ckpt.get("config", cfg)
        window_size = int(
            args.window_size
            if args.window_size is not None
            else ckpt_cfg.get("dataset", {}).get("window_size", cfg.get("dataset", {}).get("window_size", 1))
        )
        normalize_mode = args.normalize_mode or resolve_normalize_mode(ckpt_cfg)
        dataset_key = (window_size, normalize_mode)
        ds = dataset_cache.get(dataset_key)
        if ds is None:
            ds = AOASampleDataset(
                aoa_root=aoa_root,
                labels_root=labels_root,
                window_size=window_size,
                normalize_mode=normalize_mode,
            )
            dataset_cache[dataset_key] = ds
        summary, grouped = summarize_checkpoint(checkpoint_path, cfg, args, ds, indices, device)
        summary_rows.append(summary)
        action_rows.extend(grouped)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    write_csv(output_dir / "checkpoint_summary.csv", summary_rows)
    write_csv(output_dir / "action_summary.csv", action_rows)

    print(f"[diagnose] samples={len(indices)} checkpoints={len(checkpoint_paths)} output_dir={output_dir}")
    for row in summary_rows:
        print(
            "[diagnose] "
            f"checkpoint={row['checkpoint']} variance_ratio={row['variance_ratio_pred_over_target']:.4f} "
            f"mse_pred={row['mse_pred_to_target']:.6f} mse_meanpose={row['mse_meanpose_to_target']:.6f} "
            f"mean_pose_gap={row['pred_target_mean_pose_gap']:.6f}"
        )


if __name__ == "__main__":
    main()
