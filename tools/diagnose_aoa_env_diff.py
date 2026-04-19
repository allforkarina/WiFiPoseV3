from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.aoa_dataset import AOASampleDataset, sample_to_env
from train import load_config, resolve_data_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose cross-environment AoA differences and low-motion frame ratios."
    )
    parser.add_argument("--config", type=str, default="configs/windows_smoke.yaml")
    parser.add_argument("--aoa_root", type=str, default=None)
    parser.add_argument("--labels_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--samples_per_env", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diff_mode", type=str, choices=["prev_frame"], default="prev_frame")
    parser.add_argument("--motion_quantile", type=float, default=0.30)
    parser.add_argument("--max_frames_per_sequence", type=int, default=0)
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--no_save_plots", action="store_false", dest="save_plots")
    parser.set_defaults(save_plots=True)
    return parser.parse_args()


def resolve_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_roots(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    config_path = resolve_path(args.config)
    if config_path is None:
        raise ValueError("A config path is required.")
    cfg = load_config(config_path)
    aoa_root, labels_root = resolve_data_roots(cfg)
    if args.aoa_root is not None:
        aoa_root = resolve_path(args.aoa_root)
    if args.labels_root is not None:
        labels_root = resolve_path(args.labels_root)
    if aoa_root is None or labels_root is None:
        raise RuntimeError("Failed to resolve data roots.")
    return cfg, aoa_root, labels_root


def resolve_output_dir(output_arg: str | None) -> Path:
    if output_arg is not None:
        output_dir = resolve_path(output_arg)
        if output_dir is None:
            raise RuntimeError("Failed to resolve output dir.")
        return output_dir
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "tmp" / "diagnostics" / "aoa_env_diff" / timestamp


def load_normalized_frames(h5_path: Path, max_frames_per_sequence: int = 0) -> np.ndarray:
    with h5py.File(h5_path, "r") as hf:
        frames = np.asarray(hf["aoa_spectrum"], dtype=np.float32)
    if max_frames_per_sequence > 0:
        frames = frames[:max_frames_per_sequence]
    return AOASampleDataset._normalize_aoa(frames)


def build_diff_frames(norm_frames: np.ndarray, diff_mode: str) -> np.ndarray:
    if diff_mode != "prev_frame":
        raise ValueError(f"Unsupported diff_mode: {diff_mode}")
    if norm_frames.shape[0] == 0:
        return norm_frames.copy()
    prev_frames = np.concatenate([norm_frames[:1], norm_frames[:-1]], axis=0)
    return norm_frames - prev_frames


def build_stratified_samples(
    dataset: AOASampleDataset,
    samples_per_env: int,
    seed: int,
    max_frames_per_sequence: int,
) -> tuple[dict[str, list[int]], dict[str, dict[str, int]], dict[str, bool]]:
    rng = np.random.default_rng(seed)
    env_action_buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    frame_limit = max(0, int(max_frames_per_sequence))
    for idx, (action, sample, frame_idx) in enumerate(dataset.index):
        if frame_limit > 0 and frame_idx >= frame_limit:
            continue
        env_action_buckets[sample_to_env(sample)][action].append(idx)

    sampled: dict[str, list[int]] = {}
    sample_counts: dict[str, dict[str, int]] = {}
    saturated: dict[str, bool] = {}

    for env_id in sorted(env_action_buckets):
        action_to_indices = env_action_buckets[env_id]
        shuffled: dict[str, list[int]] = {}
        total_available = 0
        for action, indices in action_to_indices.items():
            copied = list(indices)
            rng.shuffle(copied)
            shuffled[action] = copied
            total_available += len(copied)

        target = min(samples_per_env, total_available)
        saturated[env_id] = total_available <= samples_per_env
        selected: list[int] = []
        selected_counts: dict[str, int] = {action: 0 for action in shuffled}
        action_cycle = sorted(shuffled)

        while len(selected) < target:
            progressed = False
            for action in action_cycle:
                bucket = shuffled[action]
                if not bucket:
                    continue
                selected.append(bucket.pop())
                selected_counts[action] += 1
                progressed = True
                if len(selected) >= target:
                    break
            if not progressed:
                break

        sampled[env_id] = selected
        sample_counts[env_id] = selected_counts

    return sampled, sample_counts, saturated


def collect_sampled_feature_rows(
    dataset: AOASampleDataset,
    sampled_indices: dict[str, list[int]],
    diff_mode: str,
    max_frames_per_sequence: int,
) -> list[dict[str, Any]]:
    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []

    for env_id in sorted(sampled_indices):
        for idx in sampled_indices[env_id]:
            action, sample, frame_idx = dataset.index[idx]
            seq_key = (action, sample)
            if seq_key not in cache:
                h5_path = dataset._h5_map[seq_key]
                norm_frames = load_normalized_frames(h5_path, max_frames_per_sequence=max_frames_per_sequence)
                diff_frames = build_diff_frames(norm_frames, diff_mode=diff_mode)
                cache[seq_key] = (norm_frames, diff_frames)
            norm_frames, diff_frames = cache[seq_key]
            if frame_idx >= norm_frames.shape[0]:
                continue
            abs_vec = norm_frames[frame_idx]
            diff_vec = diff_frames[frame_idx]
            rows.append(
                {
                    "env_id": env_id,
                    "action": action,
                    "sample": sample,
                    "frame_idx": int(frame_idx),
                    "abs_vec": abs_vec,
                    "diff_vec": diff_vec,
                    "motion_energy": float(np.mean(np.abs(diff_vec))),
                }
            )
    return rows


def collect_motion_energy_records(
    dataset: AOASampleDataset,
    diff_mode: str,
    max_frames_per_sequence: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for action, sample in sorted(dataset._h5_map):
        h5_path = dataset._h5_map[(action, sample)]
        norm_frames = load_normalized_frames(h5_path, max_frames_per_sequence=max_frames_per_sequence)
        diff_frames = build_diff_frames(norm_frames, diff_mode=diff_mode)
        env_id = sample_to_env(sample)
        motion_energy = np.mean(np.abs(diff_frames), axis=1) if diff_frames.size else np.zeros((0,), dtype=np.float32)
        for frame_idx, energy in enumerate(motion_energy):
            records.append(
                {
                    "env_id": env_id,
                    "action": action,
                    "sample": sample,
                    "frame_idx": int(frame_idx),
                    "motion_energy": float(energy),
                }
            )
    return records


def mean_vector(vectors: list[np.ndarray]) -> np.ndarray | None:
    if not vectors:
        return None
    return np.mean(np.stack(vectors, axis=0), axis=0)


def l2_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.linalg.norm(vec_a - vec_b))


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-12:
        return 0.0
    cosine_sim = float(np.dot(vec_a, vec_b) / denom)
    cosine_sim = min(1.0, max(-1.0, cosine_sim))
    return 1.0 - cosine_sim


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return math.inf if numerator > 0 else 0.0
    return numerator / denominator


def summarize_pairwise_env_distances(
    env_means_abs: dict[str, np.ndarray],
    env_means_diff: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    env_ids = sorted(set(env_means_abs) & set(env_means_diff))
    pair_rows: list[dict[str, Any]] = []
    abs_l2_values: list[float] = []
    diff_l2_values: list[float] = []
    abs_cos_values: list[float] = []
    diff_cos_values: list[float] = []

    for idx, env_a in enumerate(env_ids):
        for env_b in env_ids[idx + 1:]:
            abs_l2 = l2_distance(env_means_abs[env_a], env_means_abs[env_b])
            diff_l2 = l2_distance(env_means_diff[env_a], env_means_diff[env_b])
            abs_cos = cosine_distance(env_means_abs[env_a], env_means_abs[env_b])
            diff_cos = cosine_distance(env_means_diff[env_a], env_means_diff[env_b])
            pair_rows.append(
                {
                    "env_a": env_a,
                    "env_b": env_b,
                    "abs_l2": abs_l2,
                    "diff_l2": diff_l2,
                    "abs_cos": abs_cos,
                    "diff_cos": diff_cos,
                }
            )
            abs_l2_values.append(abs_l2)
            diff_l2_values.append(diff_l2)
            abs_cos_values.append(abs_cos)
            diff_cos_values.append(diff_cos)

    summary = {
        "abs_env_pair_l2_mean": float(np.mean(abs_l2_values)) if abs_l2_values else 0.0,
        "diff_env_pair_l2_mean": float(np.mean(diff_l2_values)) if diff_l2_values else 0.0,
        "abs_env_pair_cos_mean": float(np.mean(abs_cos_values)) if abs_cos_values else 0.0,
        "diff_env_pair_cos_mean": float(np.mean(diff_cos_values)) if diff_cos_values else 0.0,
    }
    summary["env_gap_reduction_ratio_l2"] = safe_ratio(
        summary["diff_env_pair_l2_mean"],
        summary["abs_env_pair_l2_mean"],
    )
    return pair_rows, summary


def summarize_action_distances(sample_rows: list[dict[str, Any]]) -> dict[str, float]:
    env_action_vectors: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for row in sample_rows:
        env_action_vectors[row["env_id"]][row["action"]].append(row["diff_vec"])

    per_env_means: dict[str, dict[str, np.ndarray]] = {}
    for env_id, action_map in env_action_vectors.items():
        action_means: dict[str, np.ndarray] = {}
        for action, vectors in action_map.items():
            vec = mean_vector(vectors)
            if vec is not None:
                action_means[action] = vec
        per_env_means[env_id] = action_means

    distances: list[float] = []
    pair_count = 0
    for env_id in sorted(per_env_means):
        actions = sorted(per_env_means[env_id])
        for idx, action_a in enumerate(actions):
            for action_b in actions[idx + 1:]:
                distances.append(l2_distance(per_env_means[env_id][action_a], per_env_means[env_id][action_b]))
                pair_count += 1

    return {
        "diff_action_pair_l2_mean": float(np.mean(distances)) if distances else 0.0,
        "diff_action_pair_count": float(pair_count),
    }


def summarize_motion_energy(records: list[dict[str, Any]], motion_quantile: float) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
    if not records:
        return 0.0, {"low_motion_ratio_global": 0.0}, []

    motion_quantile = min(max(float(motion_quantile), 0.0), 1.0)
    energies = np.asarray([row["motion_energy"] for row in records], dtype=np.float32)
    threshold = float(np.quantile(energies, motion_quantile))

    env_counts: dict[str, int] = defaultdict(int)
    env_low_counts: dict[str, int] = defaultdict(int)
    action_counts: dict[str, int] = defaultdict(int)
    action_low_counts: dict[str, int] = defaultdict(int)
    env_action_counts: dict[tuple[str, str], int] = defaultdict(int)
    env_action_low_counts: dict[tuple[str, str], int] = defaultdict(int)
    env_action_energy_sums: dict[tuple[str, str], float] = defaultdict(float)

    for row in records:
        env_id = row["env_id"]
        action = row["action"]
        energy = float(row["motion_energy"])
        is_low = energy <= threshold
        env_counts[env_id] += 1
        action_counts[action] += 1
        env_action_counts[(env_id, action)] += 1
        env_action_energy_sums[(env_id, action)] += energy
        if is_low:
            env_low_counts[env_id] += 1
            action_low_counts[action] += 1
            env_action_low_counts[(env_id, action)] += 1

    metrics: dict[str, float] = {
        "motion_energy_threshold": threshold,
        "low_motion_ratio_global": float(np.mean(energies <= threshold)),
    }
    for env_id in sorted(env_counts):
        metrics[f"low_motion_ratio_{env_id}"] = safe_ratio(env_low_counts[env_id], env_counts[env_id])

    breakdown_rows: list[dict[str, Any]] = []
    all_keys = sorted(env_action_counts)
    for env_id, action in all_keys:
        total = env_action_counts[(env_id, action)]
        low = env_action_low_counts[(env_id, action)]
        breakdown_rows.append(
            {
                "env_id": env_id,
                "action": action,
                "frame_count": total,
                "mean_motion_energy": env_action_energy_sums[(env_id, action)] / max(1, total),
                "low_motion_ratio": safe_ratio(low, total),
            }
        )

    return threshold, metrics, breakdown_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_env_mean_spectra(path: Path, env_means: dict[str, np.ndarray], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for env_id in sorted(env_means):
        ax.plot(np.arange(env_means[env_id].shape[0]), env_means[env_id], linewidth=2, label=env_id)
    ax.set_title(title)
    ax.set_xlabel("AoA Bin")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_distance_summary(path: Path, abs_env_gap: float, diff_env_gap: float, diff_action_gap: float) -> None:
    labels = ["abs env gap", "diff env gap", "diff action gap"]
    values = [abs_env_gap, diff_env_gap, diff_action_gap]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(labels, values, color=colors)
    ax.set_title("AoA Distance Summary")
    ax.set_ylabel("Mean L2 Distance")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_motion_histogram(path: Path, energies: np.ndarray, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(energies, bins=40, color="#4c78a8", alpha=0.85)
    ax.axvline(threshold, color="#d62728", linestyle="--", linewidth=2, label=f"threshold={threshold:.4f}")
    ax.set_title("Motion Energy Distribution")
    ax.set_xlabel("Mean |diff AoA|")
    ax.set_ylabel("Frame Count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_conclusion(summary_row: dict[str, Any]) -> tuple[str, str]:
    env_ratio = float(summary_row["env_gap_reduction_ratio_l2"])
    action_ratio = float(summary_row["action_over_env_ratio"])
    low_motion_ratio = float(summary_row["low_motion_ratio_global"])

    if env_ratio < 0.7 and action_ratio > 1.0:
        decision = "stage_3a"
        reason = "Diff AoA already suppresses environment gaps and preserves stronger action structure."
    elif env_ratio >= 0.7:
        decision = "stage_2a1"
        reason = "Diff AoA still carries substantial environment fingerprints."
    else:
        decision = "needs_review"
        reason = "Diff AoA reduces environment gaps, but action separation is not yet stronger than environment separation."

    if low_motion_ratio > 0.30:
        reason += " Low-motion frames are frequent enough to justify curriculum-learning evaluation."

    return decision, reason


def main() -> None:
    args = parse_args()
    cfg, aoa_root, labels_root = resolve_roots(args)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = AOASampleDataset(
        aoa_root=aoa_root,
        labels_root=labels_root,
        window_size=1,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is empty. aoa_root={aoa_root}")

    sampled_indices, sample_counts, saturated = build_stratified_samples(
        dataset=dataset,
        samples_per_env=max(1, int(args.samples_per_env)),
        seed=int(args.seed),
        max_frames_per_sequence=max(0, int(args.max_frames_per_sequence)),
    )
    sample_rows = collect_sampled_feature_rows(
        dataset=dataset,
        sampled_indices=sampled_indices,
        diff_mode=args.diff_mode,
        max_frames_per_sequence=max(0, int(args.max_frames_per_sequence)),
    )
    motion_records = collect_motion_energy_records(
        dataset=dataset,
        diff_mode=args.diff_mode,
        max_frames_per_sequence=max(0, int(args.max_frames_per_sequence)),
    )

    env_abs_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    env_diff_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in sample_rows:
        env_abs_vectors[row["env_id"]].append(row["abs_vec"])
        env_diff_vectors[row["env_id"]].append(row["diff_vec"])

    env_means_abs = {
        env_id: mean_vector(vectors)
        for env_id, vectors in env_abs_vectors.items()
        if mean_vector(vectors) is not None
    }
    env_means_diff = {
        env_id: mean_vector(vectors)
        for env_id, vectors in env_diff_vectors.items()
        if mean_vector(vectors) is not None
    }
    pair_rows, env_summary = summarize_pairwise_env_distances(env_means_abs, env_means_diff)
    action_summary = summarize_action_distances(sample_rows)
    _, motion_summary, breakdown_rows = summarize_motion_energy(
        records=motion_records,
        motion_quantile=args.motion_quantile,
    )

    sampled_frame_count = len(sample_rows)
    env_count = len(env_means_diff)
    summary_row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": str(resolve_path(args.config)),
        "aoa_root": str(aoa_root),
        "labels_root": str(labels_root),
        "output_dir": str(output_dir),
        "samples_per_env": int(args.samples_per_env),
        "sampled_frame_count": sampled_frame_count,
        "env_count": env_count,
        "diff_mode": args.diff_mode,
        "motion_quantile": float(args.motion_quantile),
        "max_frames_per_sequence": max(0, int(args.max_frames_per_sequence)),
        **env_summary,
        **action_summary,
        **motion_summary,
    }
    summary_row["action_over_env_ratio"] = safe_ratio(
        float(summary_row["diff_action_pair_l2_mean"]),
        float(summary_row["diff_env_pair_l2_mean"]),
    )
    summary_row["curriculum_candidate"] = bool(float(summary_row["low_motion_ratio_global"]) > 0.30)
    for env_id in sorted(sampled_indices):
        summary_row[f"sampled_count_{env_id}"] = len(sampled_indices[env_id])
        summary_row[f"sampling_saturated_{env_id}"] = bool(saturated.get(env_id, False))

    decision, reason = build_conclusion(summary_row)
    summary_row["decision"] = decision
    summary_row["decision_reason"] = reason

    summary_csv = output_dir / "summary.csv"
    breakdown_csv = output_dir / "env_action_breakdown.csv"
    pairwise_csv = output_dir / "pairwise_env_distances.csv"
    sampling_csv = output_dir / "sampling_breakdown.csv"

    write_csv(summary_csv, [summary_row])
    write_csv(breakdown_csv, breakdown_rows)
    write_csv(pairwise_csv, pair_rows)
    write_csv(
        sampling_csv,
        [
            {
                "env_id": env_id,
                "action": action,
                "sampled_count": count,
                "sampling_saturated": bool(saturated.get(env_id, False)),
            }
            for env_id, action_map in sorted(sample_counts.items())
            for action, count in sorted(action_map.items())
        ],
    )

    if args.save_plots:
        plot_env_mean_spectra(
            output_dir / "abs_env_mean_spectra.png",
            env_means_abs,
            title="Absolute AoA Mean Spectra by Environment",
            ylabel="Normalized Power",
        )
        plot_env_mean_spectra(
            output_dir / "diff_env_mean_spectra.png",
            env_means_diff,
            title="Diff AoA Mean Spectra by Environment",
            ylabel="Mean Diff Power",
        )
        plot_distance_summary(
            output_dir / "diff_action_vs_env_distance.png",
            abs_env_gap=float(summary_row["abs_env_pair_l2_mean"]),
            diff_env_gap=float(summary_row["diff_env_pair_l2_mean"]),
            diff_action_gap=float(summary_row["diff_action_pair_l2_mean"]),
        )
        plot_motion_histogram(
            output_dir / "motion_energy_histogram.png",
            energies=np.asarray([row["motion_energy"] for row in motion_records], dtype=np.float32),
            threshold=float(summary_row["motion_energy_threshold"]),
        )

    print(f"[diagnose] dataset_frames={len(dataset)} sampled_frames={sampled_frame_count} envs={env_count}")
    print(
        f"[diagnose] abs_env_gap={float(summary_row['abs_env_pair_l2_mean']):.6f} "
        f"diff_env_gap={float(summary_row['diff_env_pair_l2_mean']):.6f} "
        f"action_gap={float(summary_row['diff_action_pair_l2_mean']):.6f}"
    )
    print(
        f"[diagnose] env_gap_reduction_ratio={float(summary_row['env_gap_reduction_ratio_l2']):.6f} "
        f"action_over_env_ratio={float(summary_row['action_over_env_ratio']):.6f}"
    )
    print(
        f"[diagnose] motion_threshold={float(summary_row['motion_energy_threshold']):.6f} "
        f"low_motion_ratio={float(summary_row['low_motion_ratio_global']):.6f}"
    )
    print(f"[diagnose] decision={decision}")
    print(f"[diagnose] reason={reason}")
    print(f"[diagnose] summary_csv={summary_csv}")
    print(f"[diagnose] breakdown_csv={breakdown_csv}")
    print(f"[diagnose] pairwise_csv={pairwise_csv}")
    print(f"[diagnose] output_dir={output_dir}")


if __name__ == "__main__":
    main()
