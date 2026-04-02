from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataloader.aoa_dataset import AOASampleDataset
from train import PROJECT_ROOT, build_model, load_config, nMPJPE, resolve_data_roots, resolve_normalize_mode
from utils.set_seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on one random sample per action and visualize results.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint. Default: latest .pth in checkpoints/")
    parser.add_argument("--model_name", type=str, default=None, help="Model name override: conv1d_baseline | resnet1d")
    parser.add_argument("--aoa_cache_root", type=str, default=None)
    parser.add_argument("--labels_root", type=str, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--normalize_mode", type=str, choices=["pelvis_torso", "mean_rms"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_checkpoint(checkpoint_arg: str | None) -> Path:
    if checkpoint_arg is not None:
        checkpoint_path = Path(checkpoint_arg)
        return checkpoint_path if checkpoint_path.is_absolute() else PROJECT_ROOT / checkpoint_path

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    candidates = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")
    return candidates[0]


# -------------------------------------------------------------
# Evaluation Tools & Metrics
# -------------------------------------------------------------
def inverse_pose(normalized_pose: torch.Tensor, meta: dict[str, Any]) -> np.ndarray:
    """
    Reverse the normalization applied during dataset extraction.
    Transforms mean/rms normalized poses back to the original physical 2D plane scale for valid mpjpe calculation.
    """
    center = np.asarray(meta["pose_center"], dtype=np.float32).reshape(1, 2)    
    scale = float(np.asarray(meta["pose_scale"], dtype=np.float32).reshape(-1)[0])
    return (normalized_pose.detach().cpu().numpy() * scale) + center


def build_action_index(ds: AOASampleDataset) -> dict[str, list[int]]:
    """ Group dataset elements by actions for stratified (per-action) evaluation plot selection """
    action_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, (action, _, _) in enumerate(ds.index):
        action_to_indices[action].append(idx)
    return dict(action_to_indices)


def save_visualization(
    out_path: Path,
    action: str,
    x: torch.Tensor,
    gt_pose: np.ndarray,
    pred_pose: np.ndarray,
    meta: dict[str, Any],
    sample_nm: float,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    aoa = x.squeeze(0).detach().cpu().numpy()
    axes[0].plot(np.arange(len(aoa)), aoa, color="#1f77b4", linewidth=2)
    axes[0].set_title(f"{action} AoA Spectrum")
    axes[0].set_xlabel("AoA Bin")
    axes[0].set_ylabel("Normalized Power")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].scatter(gt_pose[:, 0], gt_pose[:, 1], c="#2ca02c", label="GT", s=28)
    axes[1].scatter(pred_pose[:, 0], pred_pose[:, 1], c="#d62728", label="Pred", s=28)
    for idx, (gx, gy) in enumerate(gt_pose):
        axes[1].text(gx, gy, str(idx), fontsize=7, color="#2ca02c")
    for idx, (px, py) in enumerate(pred_pose):
        axes[1].text(px, py, str(idx), fontsize=7, color="#d62728")
    axes[1].invert_yaxis()
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title(f"{action} Pose | nMPJPE={sample_nm:.4f}")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].legend(loc="best")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    sample_id = meta["sample"]
    frame_idx = int(meta["frame_idx"]) + 1
    fig.suptitle(f"{action} | {sample_id} | frame={frame_idx:03d}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(config_path)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    aoa_root, labels_root = resolve_data_roots(cfg, args.aoa_cache_root, args.labels_root)
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_cfg = ckpt.get("config", cfg)
    window_size = int(args.window_size if args.window_size is not None else ckpt_cfg.get("dataset", {}).get("window_size", cfg.get("dataset", {}).get("window_size", 1)))
    normalize_mode = args.normalize_mode or resolve_normalize_mode(ckpt_cfg)

    model = build_model(ckpt_cfg, device, args.model_name, window_size=window_size)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = AOASampleDataset(
        aoa_root=aoa_root,
        labels_root=labels_root,
        window_size=window_size,
        normalize_mode=normalize_mode,
    )
    action_to_indices = build_action_index(ds)
    rng = random.Random(args.seed)

    out_dir = PROJECT_ROOT / "logs" / "eval" / checkpoint_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for action in sorted(action_to_indices):
            sample_idx = rng.choice(action_to_indices[action])
            x, y, meta = ds[sample_idx]
            x_batch = x.unsqueeze(0).to(device)
            y_batch = y.unsqueeze(0).to(device)
            pred = model(x_batch)
            sample_nm = float(nMPJPE(pred, y_batch).item())

            pred_pose = inverse_pose(pred.squeeze(0), meta)
            gt_pose = inverse_pose(y, meta)

            out_path = out_dir / f"{action}_{meta['sample']}_frame{int(meta['frame_idx']) + 1:03d}.png"
            save_visualization(
                out_path=out_path,
                action=action,
                x=x,
                gt_pose=gt_pose,
                pred_pose=pred_pose,
                meta=meta,
                sample_nm=sample_nm,
            )

            summary_rows.append(
                {
                    "action": action,
                    "sample": meta["sample"],
                    "frame_idx": int(meta["frame_idx"]),
                    "env_id": meta["env_id"],
                    "nmpjpe": sample_nm,
                    "image_path": str(out_path),
                }
            )

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    mean_nm = sum(row["nmpjpe"] for row in summary_rows) / max(1, len(summary_rows))
    print(f"[eval] checkpoint={checkpoint_path}")
    print(f"[eval] normalize_mode={normalize_mode}")
    print(f"[eval] output_dir={out_dir}")
    print(f"[eval] actions={len(summary_rows)} mean_nmpjpe={mean_nm:.6f}")
    print(f"[eval] summary_csv={summary_path}")


if __name__ == "__main__":
    main()
