from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.aoa_dataset import AOASampleDataset, sample_to_env
from train import load_config, resolve_data_roots, resolve_dataset_feature_config

SUPPORTED_INPUT_MODES = ["diff", "abs", "svd_residual", "svd_residual_diff"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose AoA feature environment gaps, action separation, and low-motion-frame ratios."
    )
    parser.add_argument("--config", type=str, default="configs/windows_smoke.yaml")
    parser.add_argument("--aoa_root", type=str, default=None)
    parser.add_argument("--labels_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--samples_per_env", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_mode", type=str, choices=SUPPORTED_INPUT_MODES, default=None)
    parser.add_argument("--svd_rank", type=int, default=None)
    parser.add_argument("--compare_input_modes", type=str, default=None)
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


def resolve_analysis_modes(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], int, bool]:
    feature_cfg = resolve_dataset_feature_config(cfg)
    default_mode = args.input_mode or feature_cfg["input_mode"]
    svd_rank = max(1, int(args.svd_rank if args.svd_rank is not None else feature_cfg["svd_rank"]))
    feature_centering = bool(feature_cfg["feature_centering"])

    if args.compare_input_modes:
        requested_modes = [
            mode.strip().lower()
            for mode in args.compare_input_modes.split(",")
            if mode.strip()
        ]
    else:
        requested_modes = [default_mode]

    for mode in requested_modes:
        if mode not in SUPPORTED_INPUT_MODES:
            raise ValueError(f"Unsupported input_mode in compare_input_modes: {mode}")

    analysis_modes = ["abs"]
    for mode in requested_modes:
        if mode not in analysis_modes:
            analysis_modes.append(mode)

    return analysis_modes, svd_rank, feature_centering


def build_datasets(
    aoa_root: Path,
    labels_root: Path,
    analysis_modes: list[str],
    svd_rank: int,
    feature_centering: bool,
) -> dict[str, AOASampleDataset]:
    datasets: dict[str, AOASampleDataset] = {}
    for mode in analysis_modes:
        datasets[mode] = AOASampleDataset(
            aoa_root=aoa_root,
            labels_root=labels_root,
            window_size=1,
            normalize_mode="mean_rms",
            input_mode=mode,
            svd_rank=svd_rank,
            feature_centering=feature_centering,
            cache_in_memory=False,
        )
    return datasets


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
    max_frames_per_sequence: int,
) -> list[dict[str, Any]]:
    cache: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    for env_id in sorted(sampled_indices):
        for idx in sampled_indices[env_id]:
            action, sample, frame_idx = dataset.index[idx]
            seq_key = (action, sample)
            if seq_key not in cache:
                cache[seq_key] = dataset.load_feature_sequence(
                    action,
                    sample,
                    max_frames_per_sequence=max_frames_per_sequence,
                )
            feature_sequence = cache[seq_key]
            if frame_idx >= feature_sequence.shape[0]:
                continue
            feature_vec = feature_sequence[frame_idx]
            rows.append(
                {
                    "env_id": env_id,
                    "action": action,
                    "sample": sample,
                    "frame_idx": int(frame_idx),
                    "feature_vec": feature_vec,
                    "motion_energy": float(np.mean(np.abs(feature_vec))),
                }
            )
    return rows


def collect_motion_energy_records(
    dataset: AOASampleDataset,
    max_frames_per_sequence: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for action, sample in sorted(dataset._h5_map):
        feature_sequence = dataset.load_feature_sequence(
            action,
            sample,
            max_frames_per_sequence=max_frames_per_sequence,
        )
        env_id = sample_to_env(sample)
        motion_energy = np.mean(np.abs(feature_sequence), axis=1) if feature_sequence.size else np.zeros((0,), dtype=np.float32)
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
    env_means: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    env_ids = sorted(env_means)
    pair_rows: list[dict[str, Any]] = []
    l2_values: list[float] = []
    cos_values: list[float] = []

    for idx, env_a in enumerate(env_ids):
        for env_b in env_ids[idx + 1:]:
            pair_l2 = l2_distance(env_means[env_a], env_means[env_b])
            pair_cos = cosine_distance(env_means[env_a], env_means[env_b])
            pair_rows.append(
                {
                    "env_a": env_a,
                    "env_b": env_b,
                    "env_pair_l2": pair_l2,
                    "env_pair_cos": pair_cos,
                }
            )
            l2_values.append(pair_l2)
            cos_values.append(pair_cos)

    return pair_rows, {
        "env_pair_l2_mean": float(np.mean(l2_values)) if l2_values else 0.0,
        "env_pair_cos_mean": float(np.mean(cos_values)) if cos_values else 0.0,
    }


def summarize_action_distances(sample_rows: list[dict[str, Any]]) -> dict[str, float]:
    env_action_vectors: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for row in sample_rows:
        env_action_vectors[row["env_id"]][row["action"]].append(row["feature_vec"])

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
        "action_pair_l2_mean": float(np.mean(distances)) if distances else 0.0,
        "action_pair_count": float(pair_count),
    }


def summarize_motion_energy(
    records: list[dict[str, Any]],
    motion_quantile: float,
) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
    if not records:
        return 0.0, {"motion_energy_threshold": 0.0, "low_motion_ratio_global": 0.0}, []

    motion_quantile = min(max(float(motion_quantile), 0.0), 1.0)
    energies = np.asarray([row["motion_energy"] for row in records], dtype=np.float32)
    threshold = float(np.quantile(energies, motion_quantile))

    env_counts: dict[str, int] = defaultdict(int)
    env_low_counts: dict[str, int] = defaultdict(int)
    env_action_counts: dict[tuple[str, str], int] = defaultdict(int)
    env_action_low_counts: dict[tuple[str, str], int] = defaultdict(int)
    env_action_energy_sums: dict[tuple[str, str], float] = defaultdict(float)

    for row in records:
        env_id = row["env_id"]
        action = row["action"]
        energy = float(row["motion_energy"])
        is_low = energy <= threshold
        env_counts[env_id] += 1
        env_action_counts[(env_id, action)] += 1
        env_action_energy_sums[(env_id, action)] += energy
        if is_low:
            env_low_counts[env_id] += 1
            env_action_low_counts[(env_id, action)] += 1

    metrics: dict[str, float] = {
        "motion_energy_threshold": threshold,
        "low_motion_ratio_global": float(np.mean(energies <= threshold)),
    }
    for env_id in sorted(env_counts):
        metrics[f"low_motion_ratio_{env_id}"] = safe_ratio(env_low_counts[env_id], env_counts[env_id])

    breakdown_rows: list[dict[str, Any]] = []
    for env_id, action in sorted(env_action_counts):
        total = env_action_counts[(env_id, action)]
        breakdown_rows.append(
            {
                "env_id": env_id,
                "action": action,
                "frame_count": total,
                "mean_motion_energy": env_action_energy_sums[(env_id, action)] / max(1, total),
                "low_motion_ratio": safe_ratio(env_action_low_counts[(env_id, action)], total),
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


def plot_motion_histogram(path: Path, energies: np.ndarray, threshold: float, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(energies, bins=40, color="#4c78a8", alpha=0.85)
    ax.axvline(threshold, color="#d62728", linestyle="--", linewidth=2, label=f"threshold={threshold:.4f}")
    ax.set_title(title)
    ax.set_xlabel("Mean |feature|")
    ax.set_ylabel("Frame Count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_mode_comparison(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    if not summary_rows:
        return
    modes = [row["input_mode"] for row in summary_rows]
    env_gaps = [float(row["env_pair_l2_mean"]) for row in summary_rows]
    action_ratios = [float(row["action_over_env_ratio"]) for row in summary_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(modes, env_gaps, color="#1f77b4")
    axes[0].set_title("Environment Gap by Input Mode")
    axes[0].set_ylabel("Mean L2 Distance")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(modes, action_ratios, color="#2ca02c")
    axes[1].set_title("Action-over-Environment Ratio")
    axes[1].set_ylabel("Ratio")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_conclusion(mode: str, env_gap_ratio: float, action_ratio: float, low_motion_ratio: float) -> tuple[str, str]:
    if mode == "abs":
        return "reference", "Absolute AoA is retained as the environment-gap reference mode."

    if env_gap_ratio < 0.7 and action_ratio > 1.0:
        decision = "promote_mixed"
        reason = "This feature mode suppresses environment gaps while preserving stronger action separation."
    elif env_gap_ratio >= 0.7:
        decision = "keep_iterating"
        reason = "This feature mode still carries substantial environment fingerprints."
    else:
        decision = "needs_review"
        reason = "This feature mode reduces environment gaps, but action separation is not yet stronger than environment separation."

    if low_motion_ratio > 0.30:
        reason += " Low-motion frames are frequent enough to justify curriculum-learning evaluation."

    return decision, reason


def main() -> None:
    args = parse_args()
    cfg, aoa_root, labels_root = resolve_roots(args)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_modes, svd_rank, feature_centering = resolve_analysis_modes(cfg, args)
    datasets = build_datasets(
        aoa_root=aoa_root,
        labels_root=labels_root,
        analysis_modes=analysis_modes,
        svd_rank=svd_rank,
        feature_centering=feature_centering,
    )
    base_dataset = datasets[analysis_modes[0]]
    if len(base_dataset) == 0:
        raise RuntimeError(f"Dataset is empty. aoa_root={aoa_root}")

    max_frames_per_sequence = max(0, int(args.max_frames_per_sequence))
    sampled_indices, sample_counts, saturated = build_stratified_samples(
        dataset=base_dataset,
        samples_per_env=max(1, int(args.samples_per_env)),
        seed=int(args.seed),
        max_frames_per_sequence=max_frames_per_sequence,
    )

    mode_results: dict[str, dict[str, Any]] = {}
    pairwise_rows_all: list[dict[str, Any]] = []
    breakdown_rows_all: list[dict[str, Any]] = []

    for mode in analysis_modes:
        dataset = datasets[mode]
        sample_rows = collect_sampled_feature_rows(
            dataset=dataset,
            sampled_indices=sampled_indices,
            max_frames_per_sequence=max_frames_per_sequence,
        )
        motion_records = collect_motion_energy_records(
            dataset=dataset,
            max_frames_per_sequence=max_frames_per_sequence,
        )

        env_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
        for row in sample_rows:
            env_vectors[row["env_id"]].append(row["feature_vec"])
        env_means = {
            env_id: mean_vector(vectors)
            for env_id, vectors in env_vectors.items()
            if mean_vector(vectors) is not None
        }

        pairwise_rows, env_summary = summarize_pairwise_env_distances(env_means)
        action_summary = summarize_action_distances(sample_rows)
        _, motion_summary, breakdown_rows = summarize_motion_energy(
            records=motion_records,
            motion_quantile=args.motion_quantile,
        )

        for row in pairwise_rows:
            row["input_mode"] = mode
        for row in breakdown_rows:
            row["input_mode"] = mode

        pairwise_rows_all.extend(pairwise_rows)
        breakdown_rows_all.extend(breakdown_rows)
        mode_results[mode] = {
            "sample_rows": sample_rows,
            "motion_records": motion_records,
            "env_means": env_means,
            "env_summary": env_summary,
            "action_summary": action_summary,
            "motion_summary": motion_summary,
        }

    requested_target_modes = [mode for mode in analysis_modes if mode != "abs"] or analysis_modes
    best_env_gap_mode = min(
        requested_target_modes,
        key=lambda mode: float(mode_results[mode]["env_summary"]["env_pair_l2_mean"]),
    )
    best_action_over_env_mode = max(
        requested_target_modes,
        key=lambda mode: safe_ratio(
            float(mode_results[mode]["action_summary"]["action_pair_l2_mean"]),
            float(mode_results[mode]["env_summary"]["env_pair_l2_mean"]),
        ),
    )

    abs_reference_gap = float(mode_results["abs"]["env_summary"]["env_pair_l2_mean"])
    summary_rows: list[dict[str, Any]] = []
    for mode in analysis_modes:
        result = mode_results[mode]
        env_summary = result["env_summary"]
        action_summary = result["action_summary"]
        motion_summary = result["motion_summary"]

        action_over_env_ratio = safe_ratio(
            float(action_summary["action_pair_l2_mean"]),
            float(env_summary["env_pair_l2_mean"]),
        )
        env_gap_reduction_ratio = safe_ratio(
            float(env_summary["env_pair_l2_mean"]),
            abs_reference_gap,
        )
        decision, reason = build_conclusion(
            mode=mode,
            env_gap_ratio=env_gap_reduction_ratio,
            action_ratio=action_over_env_ratio,
            low_motion_ratio=float(motion_summary["low_motion_ratio_global"]),
        )

        summary_row: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": str(resolve_path(args.config)),
            "aoa_root": str(aoa_root),
            "labels_root": str(labels_root),
            "output_dir": str(output_dir),
            "input_mode": mode,
            "svd_rank": svd_rank,
            "feature_centering": feature_centering,
            "samples_per_env": int(args.samples_per_env),
            "sampled_frame_count": len(result["sample_rows"]),
            "env_count": len(result["env_means"]),
            "motion_quantile": float(args.motion_quantile),
            "max_frames_per_sequence": max_frames_per_sequence,
            "reference_abs_env_pair_l2_mean": abs_reference_gap,
            "env_pair_l2_mean": float(env_summary["env_pair_l2_mean"]),
            "env_pair_cos_mean": float(env_summary["env_pair_cos_mean"]),
            "env_gap_reduction_ratio_l2": env_gap_reduction_ratio,
            "action_pair_l2_mean": float(action_summary["action_pair_l2_mean"]),
            "action_pair_count": float(action_summary["action_pair_count"]),
            "motion_energy_threshold": float(motion_summary["motion_energy_threshold"]),
            "low_motion_ratio_global": float(motion_summary["low_motion_ratio_global"]),
            "action_over_env_ratio": action_over_env_ratio,
            "best_env_gap_mode": best_env_gap_mode,
            "best_action_over_env_mode": best_action_over_env_mode,
            "curriculum_candidate": bool(float(motion_summary["low_motion_ratio_global"]) > 0.30),
            "decision": decision,
            "decision_reason": reason,
        }
        for env_id in sorted(sampled_indices):
            summary_row[f"sampled_count_{env_id}"] = len(sampled_indices[env_id])
            summary_row[f"sampling_saturated_{env_id}"] = bool(saturated.get(env_id, False))
            summary_row[f"low_motion_ratio_{env_id}"] = float(motion_summary.get(f"low_motion_ratio_{env_id}", 0.0))
        summary_rows.append(summary_row)

    summary_csv = output_dir / "summary.csv"
    breakdown_csv = output_dir / "env_action_breakdown.csv"
    pairwise_csv = output_dir / "pairwise_env_distances.csv"
    sampling_csv = output_dir / "sampling_breakdown.csv"

    write_csv(summary_csv, summary_rows)
    write_csv(breakdown_csv, breakdown_rows_all)
    write_csv(pairwise_csv, pairwise_rows_all)
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
        for mode in analysis_modes:
            env_means = mode_results[mode]["env_means"]
            plot_env_mean_spectra(
                output_dir / f"{mode}_env_mean_spectra.png",
                env_means,
                title=f"{mode} Mean Spectra by Environment",
                ylabel="Mean Feature Value",
            )
            plot_motion_histogram(
                output_dir / f"{mode}_motion_energy_histogram.png",
                energies=np.asarray(
                    [row["motion_energy"] for row in mode_results[mode]["motion_records"]],
                    dtype=np.float32,
                ),
                threshold=float(mode_results[mode]["motion_summary"]["motion_energy_threshold"]),
                title=f"{mode} Motion Energy Distribution",
            )
        plot_mode_comparison(
            output_dir / "mode_comparison_distance.png",
            summary_rows,
        )

    print(
        f"[diagnose] dataset_frames={len(base_dataset)} sampled_frames={sum(len(v) for v in sampled_indices.values())} "
        f"envs={len(sampled_indices)} modes={','.join(analysis_modes)}"
    )
    for row in summary_rows:
        print(
            f"[diagnose] mode={row['input_mode']} env_gap={float(row['env_pair_l2_mean']):.6f} "
            f"env_gap_ratio={float(row['env_gap_reduction_ratio_l2']):.6f} "
            f"action_gap={float(row['action_pair_l2_mean']):.6f} "
            f"action_over_env={float(row['action_over_env_ratio']):.6f} "
            f"low_motion_ratio={float(row['low_motion_ratio_global']):.6f} decision={row['decision']}"
        )
    print(f"[diagnose] best_env_gap_mode={best_env_gap_mode}")
    print(f"[diagnose] best_action_over_env_mode={best_action_over_env_mode}")
    print(f"[diagnose] summary_csv={summary_csv}")
    print(f"[diagnose] breakdown_csv={breakdown_csv}")
    print(f"[diagnose] pairwise_csv={pairwise_csv}")
    print(f"[diagnose] output_dir={output_dir}")


if __name__ == "__main__":
    main()
