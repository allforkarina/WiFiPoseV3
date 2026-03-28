from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.aoa_dataset import AOASampleDataset


def resolve_path(path_str: str, default: Path) -> Path:
    path = Path(path_str) if path_str else default
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether AoA input distances track pose distances under different normalization modes."
    )
    parser.add_argument("--aoa_cache_root", type=str, default="data/aoa_cache")
    parser.add_argument("--labels_root", type=str, default="data/dataset")
    parser.add_argument("--label_normalize_mode", type=str, choices=["mean_rms", "pelvis_torso"], default="mean_rms")
    parser.add_argument("--window_sizes", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--aoa_modes", nargs="+", default=["raw_log", "curr_norm"])
    parser.add_argument("--max_samples", type=int, default=400)
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


def normalize_raw_aoa(frame: np.ndarray, mode: str) -> np.ndarray:
    aoa = np.asarray(frame, dtype=np.float32)
    aoa = np.nan_to_num(aoa, nan=0.0, posinf=0.0, neginf=0.0)
    aoa = np.maximum(aoa, 0.0)
    if mode == "raw":
        return aoa.astype(np.float32)
    if mode == "raw_log":
        return np.log1p(aoa).astype(np.float32)
    if mode == "curr_norm":
        return AOASampleDataset._normalize_aoa(aoa)
    raise ValueError(f"Unsupported aoa_mode: {mode}")


def load_window(ds: AOASampleDataset, idx: int, aoa_mode: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    action, sample, frame_idx = ds.index[idx]
    h5_path = ds._h5_map[(action, sample)]
    with h5py.File(h5_path, "r") as hf:
        aoa_frames = hf["aoa_spectrum"]
        num_frames = int(aoa_frames.shape[0])
        window = []
        for offset in range(-ds.window_radius, ds.window_radius + 1):
            source_idx = min(max(frame_idx + offset, 0), num_frames - 1)
            window.append(normalize_raw_aoa(np.asarray(aoa_frames[source_idx]), aoa_mode))
    x = np.stack(window, axis=0)
    _, y, meta = ds[idx]
    return x, y.numpy(), meta


def compute_pair_stats(x_arr: np.ndarray, y_arr: np.ndarray, actions: list[str]) -> dict[str, float]:
    x_dist = np.linalg.norm(x_arr[:, None, :] - x_arr[None, :, :], axis=-1)
    y_dist = np.linalg.norm(y_arr[:, None, :] - y_arr[None, :, :], axis=-1)
    tri_mask = np.triu(np.ones_like(x_dist, dtype=bool), k=1)
    x_vals = x_dist[tri_mask]
    y_vals = y_dist[tri_mask]
    corr = float(np.corrcoef(x_vals, y_vals)[0, 1]) if len(x_vals) >= 2 else float("nan")

    same_action_x: list[float] = []
    diff_action_x: list[float] = []
    same_action_y: list[float] = []
    diff_action_y: list[float] = []
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            if actions[i] == actions[j]:
                same_action_x.append(float(x_dist[i, j]))
                same_action_y.append(float(y_dist[i, j]))
            else:
                diff_action_x.append(float(x_dist[i, j]))
                diff_action_y.append(float(y_dist[i, j]))

    return {
        "pearson_xdist_ydist": corr,
        "same_action_xdist_mean": float(np.mean(same_action_x)) if same_action_x else float("nan"),
        "diff_action_xdist_mean": float(np.mean(diff_action_x)) if diff_action_x else float("nan"),
        "same_action_ydist_mean": float(np.mean(same_action_y)) if same_action_y else float("nan"),
        "diff_action_ydist_mean": float(np.mean(diff_action_y)) if diff_action_y else float("nan"),
    }


def sample_indices(ds: AOASampleDataset, max_samples: int) -> list[int]:
    step = max(1, len(ds) // max(1, max_samples))
    indices: list[int] = []
    for idx in range(0, len(ds), step):
        indices.append(idx)
        if len(indices) >= max_samples:
            break
    return indices


def main() -> None:
    args = parse_args()
    aoa_root = resolve_path(args.aoa_cache_root, PROJECT_ROOT / "data" / "aoa_cache")
    labels_root = resolve_path(args.labels_root, PROJECT_ROOT / "data" / "dataset")

    rows: list[dict[str, Any]] = []
    for window_size in args.window_sizes:
        ds = AOASampleDataset(
            aoa_root=aoa_root,
            labels_root=labels_root,
            window_size=window_size,
            normalize_mode=args.label_normalize_mode,
        )
        indices = sample_indices(ds, args.max_samples)
        for aoa_mode in args.aoa_modes:
            x_samples: list[np.ndarray] = []
            y_samples: list[np.ndarray] = []
            actions: list[str] = []
            for idx in indices:
                x, y, meta = load_window(ds, idx, aoa_mode)
                x_samples.append(x.reshape(-1))
                y_samples.append(y.reshape(-1))
                actions.append(str(meta["action"]))

            x_arr = np.stack(x_samples)
            y_arr = np.stack(y_samples)
            row = {
                "window_size": int(window_size),
                "aoa_mode": aoa_mode,
                "label_normalize_mode": args.label_normalize_mode,
                "samples": int(len(indices)),
            }
            row.update(compute_pair_stats(x_arr, y_arr, actions))
            rows.append(row)

    for row in rows:
        print(
            "[separability] "
            f"window={row['window_size']} aoa_mode={row['aoa_mode']} label_mode={row['label_normalize_mode']} "
            f"samples={row['samples']} pearson={row['pearson_xdist_ydist']:.6f} "
            f"same_x={row['same_action_xdist_mean']:.6f} diff_x={row['diff_action_xdist_mean']:.6f} "
            f"same_y={row['same_action_ydist_mean']:.6f} diff_y={row['diff_action_ydist_mean']:.6f}"
        )

    if args.output_csv:
        out_path = resolve_path(args.output_csv, PROJECT_ROOT / "logs" / "diagnose_input_pose_separability.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[separability] wrote csv={out_path}")


if __name__ == "__main__":
    main()
