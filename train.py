from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

from dataloader.aoa_dataset import AOASampleDataset, sample_to_env
from dataloader.stratified_sampler import StratifiedBatchSampler
from mymodels import ResNet1DPose
from utils.set_seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parent
ACTION_NAMES = [f"A{idx:02d}" for idx in range(1, 28)]


def load_config(cfg_path: str | Path) -> Dict[str, Any]:
	with open(cfg_path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)


class TrainLogger:
	def __init__(self, log_path: Path | None, verbose: bool) -> None:
		self.log_path = log_path
		self.verbose = verbose
		if self.log_path is not None:
			self.log_path.parent.mkdir(parents=True, exist_ok=True)

	def log(self, msg: str, *, always: bool = False) -> None:
		if self.verbose or always:
			print(msg, flush=True)
		if self.log_path is not None:
			with self.log_path.open("a", encoding="utf-8") as f:
				f.write(msg + "\n")


def nMPJPE(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
	batch_size = pred.shape[0]
	pred_flat = pred.view(batch_size, -1)
	gt_flat = gt.view(batch_size, -1)
	num = (pred_flat * gt_flat).sum(dim=1)
	den = (pred_flat * pred_flat).sum(dim=1).clamp(min=1e-8)
	scale = (num / den).view(batch_size, 1, 1)
	aligned = scale * pred
	err = torch.linalg.norm(aligned - gt, dim=-1)
	return err.mean()


class PoseStructureLoss(nn.Module):
	def __init__(
		self,
		lambda_pose: float = 1.0,
		lambda_dist: float = 0.5,
		lambda_var: float = 0.1,
		preserve_ratio: float = 0.6,
		huber_beta: float = 0.05,
	) -> None:
		super().__init__()
		self.lambda_pose = float(lambda_pose)
		self.lambda_dist = float(lambda_dist)
		self.lambda_var = float(lambda_var)
		self.preserve_ratio = float(preserve_ratio)
		self.huber_beta = float(huber_beta)

	def forward(
		self,
		pred: torch.Tensor,
		target: torch.Tensor,
	) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
		pose_loss = F.smooth_l1_loss(pred, target, beta=self.huber_beta)

		pred_dist = torch.cdist(pred, pred, p=2)
		target_dist = torch.cdist(target, target, p=2)
		dist_loss = F.l1_loss(pred_dist, target_dist)

		pred_flat = pred.view(pred.size(0), -1)
		target_flat = target.view(target.size(0), -1)
		pred_std = pred_flat.std(dim=0, unbiased=False)
		target_std = target_flat.std(dim=0, unbiased=False)
		
		var_loss = F.relu((target_std * self.preserve_ratio) - pred_std).mean()
		std_ratio = pred_std.mean() / target_std.mean().clamp(min=1e-8)

		total = (
			self.lambda_pose * pose_loss
			+ self.lambda_dist * dist_loss
			+ self.lambda_var * var_loss
		)
		parts = {
			"pose_loss": pose_loss.detach(),
			"dist_loss": dist_loss.detach(),
			"var_loss": var_loss.detach(),
			"std_ratio": std_ratio.detach(),
		}
		return total, parts


class ActionAuxHead(nn.Module):
	def __init__(self, input_dim: int, num_actions: int, dropout: float = 0.1) -> None:
		super().__init__()
		hidden_dim = max(64, input_dim // 2)
		self.net = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, num_actions),
		)

	def forward(self, feats: torch.Tensor) -> torch.Tensor:
		return self.net(feats)


def resolve_path(path_value: str | Path | None, default: Path) -> Path:
	if path_value is None:
		return default
	path = Path(path_value)
	return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_data_roots(cfg: Dict[str, Any]) -> tuple[Path, Path]:
	data_root = resolve_path(cfg.get("data_root"), PROJECT_ROOT / "data")
	aoa_root = resolve_path(cfg.get("aoa_cache_root"), data_root / "aoa_cache")
	labels_root = resolve_path(cfg.get("labels_root"), data_root / "dataset")
	return aoa_root, labels_root


def resolve_normalize_mode(cfg: Dict[str, Any]) -> str:
	loss_cfg = cfg.get("loss", {})
	return str(loss_cfg.get("normalize_mode", "pelvis_torso")).strip().lower() or "pelvis_torso"


def resolve_dataset_feature_config(cfg: Dict[str, Any]) -> dict[str, Any]:
	dataset_cfg = cfg.get("dataset", {})
	return {
		"input_mode": str(dataset_cfg.get("input_mode", "diff")).strip().lower() or "diff",
		"svd_rank": max(1, int(dataset_cfg.get("svd_rank", 1))),
		"feature_centering": bool(dataset_cfg.get("feature_centering", False)),
		"cache_in_memory": bool(dataset_cfg.get("cache_in_memory", False)),
	}


def compute_checkpoint_selection_score(
	val_nmpjpe: float,
	val_std_ratio: float,
	checkpoint_cfg: dict[str, Any],
) -> tuple[float, str]:
	selection_cfg = checkpoint_cfg.get("selection", {})
	mode = str(selection_cfg.get("mode", "accuracy")).lower()
	if mode == "accuracy":
		return -float(val_nmpjpe), "accuracy"

	if mode == "diversity_first":
		std_w = float(selection_cfg.get("std_ratio_weight", 0.35))
		nm_penalty = float(selection_cfg.get("nmpjpe_penalty", 0.02))
		score = std_w * val_std_ratio - nm_penalty * val_nmpjpe
		return score, "diversity_first"

	if mode == "balanced":
		std_floor = float(selection_cfg.get("std_floor", 0.8))
		std_bonus_weight = float(selection_cfg.get("std_bonus_weight", 0.02))
		std_penalty_weight = float(selection_cfg.get("std_penalty_weight", 0.20))
		bonus = std_bonus_weight * min(val_std_ratio, std_floor)
		penalty = std_penalty_weight * max(0.0, std_floor - val_std_ratio)
		score = -float(val_nmpjpe) + bonus - penalty
		return score, "balanced"

	return -float(val_nmpjpe), mode


def extract_meta_field(meta: Any, key: str) -> list[str]:
	if isinstance(meta, dict):
		value = meta.get(key, [])
		if isinstance(value, (list, tuple)):
			return [str(item) for item in value]
		return [str(value)]
	if isinstance(meta, (list, tuple)):
		values: list[str] = []
		for item in meta:
			if isinstance(item, dict):
				values.append(str(item.get(key, "")))
		return values
	return []


def action_to_index(action_id: str) -> int:
	if action_id in ACTION_NAMES:
		return ACTION_NAMES.index(action_id)
	try:
		return max(0, int(action_id.lstrip("A").lstrip("a")) - 1)
	except Exception:
		return 0


def build_action_targets(meta: Any, device: torch.device) -> torch.Tensor:
	actions = extract_meta_field(meta, "action")
	return torch.tensor([action_to_index(action) for action in actions], dtype=torch.long, device=device)


def resolve_optional_batch_limit(value: Any) -> int | None:
	try:
		limit = int(value)
	except (TypeError, ValueError):
		return None
	return limit if limit > 0 else None


class AOAAugmentation:
	def __init__(
		self,
		prob_noise: float = 0.0,
		noise_std: float = 0.0,
		prob_gain: float = 0.0,
		gain_range: tuple[float, float] = (1.0, 1.0),
		prob_bin_dropout: float = 0.0,
		max_dropout_bins: int = 0,
		aoa_shift_bins: int = 0,
	) -> None:
		self.prob_noise = float(prob_noise)
		self.noise_std = float(noise_std)
		self.prob_gain = float(prob_gain)
		self.gain_range = (float(gain_range[0]), float(gain_range[1]))
		self.prob_bin_dropout = float(prob_bin_dropout)
		self.max_dropout_bins = max(0, int(max_dropout_bins))
		self.aoa_shift_bins = max(0, int(aoa_shift_bins))

	def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		aug_x = x.clone()
		if self.prob_gain > 0.0 and torch.rand(()) < self.prob_gain:
			gain = torch.empty((), dtype=aug_x.dtype, device=aug_x.device).uniform_(*self.gain_range)
			aug_x = aug_x * gain
		if self.prob_noise > 0.0 and self.noise_std > 0.0 and torch.rand(()) < self.prob_noise:
			aug_x = aug_x + torch.randn_like(aug_x) * self.noise_std
		if self.prob_bin_dropout > 0.0 and self.max_dropout_bins > 0 and torch.rand(()) < self.prob_bin_dropout:
			max_bins = min(self.max_dropout_bins, int(aug_x.size(-1)))
			if max_bins > 0:
				drop_width = int(torch.randint(1, max_bins + 1, (1,)).item())
				start = int(torch.randint(0, aug_x.size(-1) - drop_width + 1, (1,)).item())
				aug_x[..., start:start + drop_width] = 0.0
		if self.aoa_shift_bins > 0:
			shift = int(torch.randint(-self.aoa_shift_bins, self.aoa_shift_bins + 1, (1,)).item())
			if shift != 0:
				aug_x = torch.roll(aug_x, shifts=shift, dims=-1)
		return aug_x, y


class TransformingDataset(Dataset):
	def __init__(
		self,
		base_dataset: Dataset,
		transform: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
	) -> None:
		self.base_dataset = base_dataset
		self.transform = transform

	def __len__(self) -> int:
		return len(self.base_dataset)

	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, Any]:
		x, y, meta = self.base_dataset[idx]
		if self.transform is not None:
			x, y = self.transform(x, y)
		return x, y, meta


def build_train_transform(cfg: Dict[str, Any]) -> AOAAugmentation | None:
	aug_cfg = cfg.get("augmentation", {})
	if not bool(aug_cfg.get("runtime_enable", False)):
		return None
	gain_range = aug_cfg.get("gain_range", [1.0, 1.0])
	if not isinstance(gain_range, (list, tuple)) or len(gain_range) != 2:
		gain_range = [1.0, 1.0]
	return AOAAugmentation(
		prob_noise=float(aug_cfg.get("prob_noise", 0.0)),
		noise_std=float(aug_cfg.get("noise_std", 0.0)),
		prob_gain=float(aug_cfg.get("prob_gain", 0.0)),
		gain_range=(float(gain_range[0]), float(gain_range[1])),
		prob_bin_dropout=float(aug_cfg.get("prob_bin_dropout", 0.0)),
		max_dropout_bins=int(aug_cfg.get("max_dropout_bins", 0)),
		aoa_shift_bins=int(aug_cfg.get("aoa_shift_bins", 0)),
	)


def forward_pose_with_features(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	if not hasattr(model, "forward_features") or not hasattr(model, "forward_head"):
		raise AttributeError("Model must implement forward_features() and forward_head() for anti-collapse training.")
	features = model.forward_features(x)
	pred = model.forward_head(features)
	return pred, features


def split_indices_by_envs(
        ds: AOASampleDataset,
        train_envs: list[str],
        val_envs: list[str],
        test_envs: list[str],
) -> tuple[list[int], list[int], list[int]]:
        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []

        for idx, (_, sample, _) in enumerate(ds.index):
                env_id = sample_to_env(sample)
                if env_id in val_envs:
                        val_indices.append(idx)
                elif env_id in test_envs:
                        test_indices.append(idx)
                else:
                        if not train_envs or env_id in train_envs:
                                train_indices.append(idx)
        return train_indices, val_indices, test_indices


def split_indices_by_action_env_sequence(
	ds: AOASampleDataset,
	train_ratio: int,
	val_ratio: int,
	test_ratio: int,
	seed: int,
) -> tuple[list[int], list[int], list[int]]:
	sequence_to_indices: dict[tuple[str, str], list[int]] = {}
	grouped_sequences: dict[tuple[str, str], list[tuple[str, str]]] = {}
	ratio_sum = train_ratio + val_ratio + test_ratio
	if ratio_sum <= 0:
		raise ValueError("Sequence split ratios must sum to a positive value.")

	for idx, (action, sample, _) in enumerate(ds.index):
		seq_key = (action, sample)
		if seq_key not in sequence_to_indices:
			sequence_to_indices[seq_key] = []
			group_key = (action, sample_to_env(sample))
			grouped_sequences.setdefault(group_key, []).append(seq_key)
		sequence_to_indices[seq_key].append(idx)

	rng = random.Random(seed)
	train_indices: list[int] = []
	val_indices: list[int] = []
	test_indices: list[int] = []

	for group_key in sorted(grouped_sequences):
		sequence_keys = sorted(grouped_sequences[group_key], key=lambda item: item[1])
		rng.shuffle(sequence_keys)
		total = len(sequence_keys)
		train_end = (total * train_ratio) // ratio_sum
		val_end = (total * (train_ratio + val_ratio)) // ratio_sum

		for seq_key in sequence_keys[:train_end]:
			train_indices.extend(sequence_to_indices[seq_key])
		for seq_key in sequence_keys[train_end:val_end]:
			val_indices.extend(sequence_to_indices[seq_key])
		for seq_key in sequence_keys[val_end:]:
			test_indices.extend(sequence_to_indices[seq_key])

	return train_indices, val_indices, test_indices


def count_unique_sequences(ds: AOASampleDataset, indices: list[int]) -> int:
	return len({(ds.index[idx][0], ds.index[idx][1]) for idx in indices})


def build_subset_loader(
        ds: AOASampleDataset,
        indices: list[int],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        shuffle: bool,
        transform: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> DataLoader:
	subset = Subset(ds, indices)
	dataset: Dataset = TransformingDataset(subset, transform) if transform is not None else subset
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


def build_dataloaders(
	cfg: Dict[str, Any],
	device: torch.device,
	aoa_root: Path,
	labels_root: Path,
	window_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
	feature_cfg = resolve_dataset_feature_config(cfg)
	ds = AOASampleDataset(
		aoa_root=aoa_root,
		labels_root=labels_root,
		window_size=window_size,
		normalize_mode=resolve_normalize_mode(cfg),
		input_mode=feature_cfg["input_mode"],
		svd_rank=feature_cfg["svd_rank"],
		feature_centering=feature_cfg["feature_centering"],
		cache_in_memory=feature_cfg["cache_in_memory"],
	)
	if len(ds) == 0:
		raise RuntimeError(f"Dataset is empty. aoa_root={aoa_root} labels_root={labels_root}")

	dataset_cfg = cfg.get("dataset", {})
	batch_size = int(dataset_cfg.get("batch_size", 32))
	num_workers = int(dataset_cfg.get("num_workers", 0))
	pin_memory = bool(dataset_cfg.get("pin_memory", False)) and device.type == "cuda"
	train_transform = build_train_transform(cfg)
	split_cfg = cfg.get("splits", {})
	protocol = str(split_cfg.get("protocol", "LOEO")).strip().lower()

	if protocol == "mixed_action_env_sequence":
		raw_ratio = split_cfg.get("sequence_ratio", [7, 2, 1])
		if not isinstance(raw_ratio, (list, tuple)) or len(raw_ratio) != 3:
			raw_ratio = [7, 2, 1]
		train_ratio, val_ratio, test_ratio = [max(0, int(value)) for value in raw_ratio]
		split_seed = int(split_cfg.get("seed", cfg.get("train", {}).get("seed", 42)))
		train_indices, val_indices, test_indices = split_indices_by_action_env_sequence(
			ds,
			train_ratio=train_ratio,
			val_ratio=val_ratio,
			test_ratio=test_ratio,
			seed=split_seed,
		)
	else:
		train_envs = dataset_cfg.get("train_envs", [])
		val_envs = dataset_cfg.get("val_envs", ["env3"])
		test_envs = dataset_cfg.get("test_envs", ["env4"])
		
		if isinstance(train_envs, str): train_envs = [train_envs]
		if isinstance(val_envs, str): val_envs = [val_envs]
		if isinstance(test_envs, str): test_envs = [test_envs]

		train_indices, val_indices, test_indices = split_indices_by_envs(ds, train_envs, val_envs, test_envs)
	if not train_indices:
		raise RuntimeError("Training split is empty")

	train_envs_seen = sorted(list(set(sample_to_env(ds.index[idx][1]) for idx in train_indices)))
	train_loader = build_subset_loader(ds, train_indices, batch_size, num_workers, pin_memory, True, train_transform)
	val_loader = build_subset_loader(ds, val_indices, batch_size, num_workers, pin_memory, False)
	test_loader = build_subset_loader(ds, test_indices, batch_size, num_workers, pin_memory, False)
	
	stats = {
		"split_protocol": protocol,
		"dataset_size": len(ds),
		"train_size": len(train_indices),
		"val_size": len(val_indices),
		"test_size": len(test_indices),
		"train_sequences": count_unique_sequences(ds, train_indices),
		"val_sequences": count_unique_sequences(ds, val_indices),
		"test_sequences": count_unique_sequences(ds, test_indices),
		"batch_size": batch_size,
		"window_size": window_size,
		"input_mode": feature_cfg["input_mode"],
		"svd_rank": feature_cfg["svd_rank"],
		"feature_centering": feature_cfg["feature_centering"],
		"cache_in_memory": feature_cfg["cache_in_memory"],
		"train_envs": train_envs_seen,
		"train_augmentation": train_transform is not None,
	}
	return train_loader, val_loader, test_loader, stats


def build_model(cfg: Dict[str, Any], device: torch.device, window_size: int | None = None, num_envs: int = 0) -> nn.Module:
	mcfg = cfg.get("model", {})
	model_name = mcfg.get("name", "resnet1d").lower()
	if model_name != "resnet1d":
		raise ValueError(f"Only resnet1d is supported, but got: {model_name}")
	effective_window = int(window_size if window_size is not None else cfg.get("dataset", {}).get("window_size", 1))
	common_kwargs = {
		"input_channels": effective_window,
		"input_length": mcfg.get("input_length", 181),
		"hidden_dim": mcfg.get("hidden_dim", 256),
		"num_joints": mcfg.get("output_joints", 17),
		"out_dim": mcfg.get("output_dim", 2),
		"dropout": mcfg.get("dropout", 0.2),
		"num_envs": num_envs,
	}
	model = ResNet1DPose(**common_kwargs)
	return model.to(device)


def resolve_epoch_lr(train_cfg: Dict[str, Any], epoch_idx: int) -> float:
	base_lr = float(train_cfg.get("lr", 3e-4))
	scheduler_cfg = train_cfg.get("lr_scheduler", {})
	if not bool(scheduler_cfg.get("enable", False)):
		return base_lr

	total_epochs = max(1, int(train_cfg.get("epochs", 1)))
	warmup_epochs = max(0, int(scheduler_cfg.get("warmup_epochs", 0)))
	min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.1))
	name = str(scheduler_cfg.get("name", "cosine")).strip().lower()

	if warmup_epochs > 0 and epoch_idx < warmup_epochs:
		return base_lr * float(epoch_idx + 1) / float(warmup_epochs)

	if name == "cosine":
		remaining_epochs = max(1, total_epochs - warmup_epochs - 1)
		progress = float(max(0, epoch_idx - warmup_epochs)) / float(remaining_epochs)
		progress = min(max(progress, 0.0), 1.0)
		scale = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
		return base_lr * scale

	return base_lr


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
	for param_group in optimizer.param_groups:
		param_group["lr"] = float(lr)


def resolve_early_stop_config(train_cfg: Dict[str, Any]) -> dict[str, Any]:
	early_cfg = train_cfg.get("early_stop", {})
	return {
		"enable": bool(early_cfg.get("enable", False)),
		"patience": max(1, int(early_cfg.get("patience", 5))),
		"min_epochs": max(1, int(early_cfg.get("min_epochs", 1))),
		"min_delta": float(early_cfg.get("min_delta", 0.0)),
	}


def save_checkpoint(
	path: Path,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	cfg: Dict[str, Any],
	extra_modules: dict[str, nn.Module] | None = None,
) -> None:
	state = {
		"model_state": model.state_dict(),
		"optimizer_state": optimizer.state_dict(),
		"epoch": epoch,
		"config": cfg,
	}
	if extra_modules:
		state["extra_modules"] = {name: module.state_dict() for name, module in extra_modules.items() if module is not None}
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(state, path)


def load_checkpoint(
	path: Path,
	model: nn.Module,
	optimizer: torch.optim.Optimizer | None = None,
	extra_modules: dict[str, nn.Module] | None = None,
) -> int:
	ckpt = torch.load(path, map_location="cpu")
	model.load_state_dict(ckpt["model_state"])
	if optimizer is not None and "optimizer_state" in ckpt:
		optimizer.load_state_dict(ckpt["optimizer_state"])
	if extra_modules and "extra_modules" in ckpt:
		for name, module in extra_modules.items():
			if module is not None and name in ckpt["extra_modules"]:
				module.load_state_dict(ckpt["extra_modules"][name])
	return int(ckpt.get("epoch", 0))


def _compute_grad_norm(*modules: nn.Module | None) -> float:
	total_sq = 0.0
	for module in modules:
		if module is None:
			continue
		for p in module.parameters():
			if p.grad is None:
				continue
			norm = p.grad.data.norm(2).item()
			total_sq += norm * norm
	return math.sqrt(total_sq) if total_sq > 0 else 0.0


def train_one_epoch(
	model: nn.Module,
	action_head: ActionAuxHead | None,
	loader: DataLoader,
	device: torch.device,
	optimizer: torch.optim.Optimizer,
	criterion: PoseStructureLoss,
	lambda_action_cls: float,
	epoch: int,
	num_epochs: int,
	logger: TrainLogger,
	log_interval: int,
	use_dann: bool = False,
	lambda_domain: float = 0.0,
	env_to_id: dict[str, int] | None = None,
	max_batches: int | None = None,
) -> tuple[float, Dict[str, float]]:
	model.train()
	if action_head is not None:
		action_head.train()
	
	running_loss = 0.0
	running_pose_loss = 0.0
	running_std_ratio = 0.0
	running_action_cls_loss = 0.0
	running_action_acc = 0.0
	running_domain_loss = 0.0
	running_domain_acc = 0.0
	
	grad_norms: list[float] = []
	total_steps = len(loader)
	effective_steps = min(total_steps, max_batches) if max_batches is not None else total_steps
	import math

	for step, (x, y, meta) in enumerate(loader):
		if max_batches is not None and step >= max_batches:
			break
		x, y = x.to(device), y.to(device)
		action_targets = build_action_targets(meta, device)
		optimizer.zero_grad()
		pred, features = forward_pose_with_features(model, x)
		
		loss, loss_parts = criterion(pred, y)
		if action_head is not None:
			action_logits = action_head(features)
			action_cls_loss = F.cross_entropy(action_logits, action_targets)
			action_acc = (action_logits.argmax(dim=1) == action_targets).float().mean()
			loss = loss + (lambda_action_cls * action_cls_loss)
		else:
			action_cls_loss = loss.detach().new_zeros(())
			action_acc = loss.detach().new_zeros(())
			
		domain_loss = loss.detach().new_zeros(())
		domain_acc = loss.detach().new_zeros(())
		if use_dann and env_to_id is not None and hasattr(model, "forward_env"):
			try:
				env_ids = extract_meta_field(meta, "env_id")
				env_targets = torch.tensor([env_to_id.get(eid, 0) for eid in env_ids], dtype=torch.long, device=device)
				p = ((epoch - 1) * effective_steps + step) / max(1, num_epochs * effective_steps)
				alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
				domain_logits = model.forward_env(features, alpha)
				domain_loss = F.cross_entropy(domain_logits, env_targets)
				domain_acc = (domain_logits.argmax(dim=1) == env_targets).float().mean()
				loss = loss + (lambda_domain * domain_loss)
			except Exception as e:
				pass

		loss.backward()
		grad_norm = _compute_grad_norm(model, action_head)
		
		log_msg = f"[train] epoch={epoch}/{num_epochs} step={step+1}/{effective_steps} loss={loss.item():.6f} grad_norm={grad_norm:.3f}"
		if action_head is not None:
			log_msg += f" acc={action_acc.item():.3f}"
		if use_dann:
			log_msg += f" d_loss={domain_loss.item():.4f} d_acc={domain_acc.item():.3f}"
		logger.log(log_msg, always=((step + 1) % log_interval == 0 or step == 0))
		
		optimizer.step()

		grad_norms.append(grad_norm)
		running_loss += loss.item()
		running_pose_loss += float(loss_parts["pose_loss"].item())
		running_std_ratio += float(loss_parts["std_ratio"].item())
		running_action_cls_loss += float(action_cls_loss.item())
		running_action_acc += float(action_acc.item())
		running_domain_loss += float(domain_loss.item())
		running_domain_acc += float(domain_acc.item())

	actual_steps = max(1, min(total_steps, max_batches) if max_batches is not None else total_steps)
	avg_loss = running_loss / actual_steps
	stats = {
		"pose_loss": running_pose_loss / actual_steps,
		"std_ratio": running_std_ratio / actual_steps,
		"action_cls_loss": running_action_cls_loss / actual_steps,
		"action_acc": running_action_acc / actual_steps,
		"domain_loss": running_domain_loss / actual_steps,
		"domain_acc": running_domain_acc / actual_steps,
		"grad_norm": float(sum(grad_norms) / max(1, len(grad_norms))),
	}
	return avg_loss, stats


def evaluate(
	model: nn.Module,
	action_head: ActionAuxHead | None,
	loader: DataLoader,
	device: torch.device,
	criterion: PoseStructureLoss,
	lambda_action_cls: float,
	use_dann: bool = False,
	env_to_id: dict[str, int] | None = None,
	max_batches: int | None = None,
) -> tuple[float, float, dict[str, float]]:
	model.eval()
	if action_head is not None:
		action_head.eval()
	
	total_loss = 0.0
	total_err = 0.0
	total_cnt = 0
	
	total_pose = 0.0
	total_std_ratio = 0.0
	total_action_cls = 0.0
	total_action_acc = 0.0
	total_domain_loss = 0.0
	total_domain_acc = 0.0

	with torch.no_grad():
		import math
		for step, (x, y, meta) in enumerate(loader):
			if max_batches is not None and step >= max_batches:
				break
			x, y = x.to(device), y.to(device)
			action_targets = build_action_targets(meta, device)
			pred, features = forward_pose_with_features(model, x)
			loss, loss_parts = criterion(pred, y)
			
			if action_head is not None:
				action_logits = action_head(features)
				action_cls_loss = F.cross_entropy(action_logits, action_targets)
				action_acc = (action_logits.argmax(dim=1) == action_targets).float().mean()
				loss = loss + (lambda_action_cls * action_cls_loss)
			else:
				action_cls_loss = loss.detach().new_zeros(())
				action_acc = loss.detach().new_zeros(())

			domain_loss = loss.detach().new_zeros(())
			domain_acc = loss.detach().new_zeros(())
			if use_dann and env_to_id is not None and hasattr(model, "forward_env"):
				try:
					env_ids = extract_meta_field(meta, "env_id")
					env_targets = torch.tensor([env_to_id.get(eid, 0) for eid in env_ids], dtype=torch.long, device=device)
					domain_logits = model.forward_env(features, 1.0)
					domain_loss = F.cross_entropy(domain_logits, env_targets)
					domain_acc = (domain_logits.argmax(dim=1) == env_targets).float().mean()
				except Exception as e:
					pass
				
			err = nMPJPE(pred, y)
			batch_size = x.size(0)
			
			total_loss += loss.item() * batch_size
			total_pose += float(loss_parts["pose_loss"].item()) * batch_size
			total_std_ratio += float(loss_parts["std_ratio"].item()) * batch_size
			total_action_cls += float(action_cls_loss.item()) * batch_size
			total_action_acc += float(action_acc.item()) * batch_size
			total_domain_loss += float(domain_loss.item()) * batch_size
			total_domain_acc += float(domain_acc.item()) * batch_size
			total_err += err.item() * batch_size
			total_cnt += batch_size

	parts = {
		"pose_loss": total_pose / max(1, total_cnt),
		"std_ratio": total_std_ratio / max(1, total_cnt),
		"action_cls_loss": total_action_cls / max(1, total_cnt),
		"action_acc": total_action_acc / max(1, total_cnt),
		"domain_loss": total_domain_loss / max(1, total_cnt),
		"domain_acc": total_domain_acc / max(1, total_cnt),
	}
	return total_loss / max(1, total_cnt), total_err / max(1, total_cnt), parts


def save_history_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
	rows = list(rows)
	if not rows:
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = list(rows[0].keys())
	with path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def assess_fit(history_rows: list[dict[str, Any]], final_test_nmpjpe: float) -> tuple[str, str]:
	if not history_rows:
		return "unknown", "No training history available."
	best_val = min(row["val_nmpjpe"] for row in history_rows)
	final_val = history_rows[-1]["val_nmpjpe"]
	final_train = history_rows[-1]["train_loss"]
	if final_train > 0.08 and final_val > 0.30:
		return "underfit", f"Train loss {final_train:.4f} and val nMPJPE {final_val:.4f} are both high."
	return "usable", f"Best Val: {best_val:.4f}. Final Test: {final_test_nmpjpe:.4f}."


def summarize_overfit_history(history_rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
	if not history_rows:
		return {
			"best_val_epoch": None,
			"best_val_nmpjpe": None,
			"first_degradation_epoch": None,
			"final_val_gap": 0.0,
			"epochs_since_best": 0,
		}
	best_row = min(history_rows, key=lambda row: float(row["val_nmpjpe"]))
	final_row = history_rows[-1]
	first_degradation_epoch = None
	for row in history_rows:
		if float(row.get("val_gap_from_best", 0.0)) > 0.0:
			first_degradation_epoch = int(row["epoch"])
			break
	return {
		"best_val_epoch": int(best_row["epoch"]),
		"best_val_nmpjpe": float(best_row["val_nmpjpe"]),
		"first_degradation_epoch": first_degradation_epoch,
		"final_val_gap": max(0.0, float(final_row["val_nmpjpe"]) - float(best_row["val_nmpjpe"])),
		"epochs_since_best": int(final_row["epoch"]) - int(best_row["epoch"]),
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Cleaned Training loop for AoA pose models")
	parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to default.yaml")
	parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
	parser.add_argument("--experiment_name", type=str, default=None, help="Optional override for run name")
	args = parser.parse_args()

	config_path = Path(args.config)
	if not config_path.is_absolute():
		config_path = PROJECT_ROOT / config_path
	cfg = load_config(config_path)
	
	train_cfg = cfg.get("train", {})
	logging_cfg = cfg.get("logging", {})
	if args.experiment_name:
		logging_cfg["run_name"] = args.experiment_name
		
	seed = int(train_cfg.get("seed", 42))
	set_seed(seed)
	window_size = int(cfg.get("dataset", {}).get("window_size", 1))
	if window_size % 2 == 0:
		window_size += 1

	device_str = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device(device_str)
	aoa_root, labels_root = resolve_data_roots(cfg)

	logs_root = resolve_path(cfg.get("logs_root"), PROJECT_ROOT / "logs")
	log_dir = logs_root / "train"
	run_name = logging_cfg.get("run_name", "default")
	model_name = cfg.get("model", {}).get("name", "resnet1d").lower()
	
	ts = datetime.now().strftime("%Y%m%d-%H%M%S")
	log_path = log_dir / f"{run_name}_{model_name}_{ts}.log"
	history_path = log_dir / f"{run_name}_{model_name}_{ts}_history.csv"
	logger = TrainLogger(log_path=log_path, verbose=bool(logging_cfg.get("verbose", True)))

	logger.log(f"[run] config={args.config} device={device} model={model_name}", always=True)
	logger.log(f"[run] window_size={window_size} normalize_mode={resolve_normalize_mode(cfg)}", always=True)

	debug_cfg = cfg.get("debug", {})
	max_train_batches = resolve_optional_batch_limit(debug_cfg.get("max_train_batches"))
	max_val_batches = resolve_optional_batch_limit(debug_cfg.get("max_val_batches"))
	max_train_eval_batches = resolve_optional_batch_limit(debug_cfg.get("max_train_eval_batches"))
	max_test_batches = resolve_optional_batch_limit(debug_cfg.get("max_test_batches"))
	skip_artifact_prune = bool(debug_cfg.get("skip_artifact_prune", False))
	if any(limit is not None for limit in (max_train_batches, max_val_batches, max_train_eval_batches, max_test_batches)):
		logger.log(
			f"[debug] max_train_batches={max_train_batches} max_val_batches={max_val_batches} "
			f"max_train_eval_batches={max_train_eval_batches} max_test_batches={max_test_batches}",
			always=True,
		)
	if skip_artifact_prune:
		logger.log("[debug] artifact pruning disabled for this run", always=True)

	train_loader, val_loader, test_loader, split_stats = build_dataloaders(
		cfg=cfg, device=device, aoa_root=aoa_root, labels_root=labels_root, window_size=window_size
	)
	logger.log(
		f"[split] protocol={split_stats['split_protocol']} "
		f"train_seq={split_stats['train_sequences']} val_seq={split_stats['val_sequences']} "
		f"test_seq={split_stats['test_sequences']}",
		always=True,
	)
	logger.log(f"[data] train={split_stats['train_size']} val={split_stats['val_size']} test={split_stats['test_size']}", always=True)
	logger.log(
		f"[feature] input_mode={split_stats['input_mode']} svd_rank={split_stats['svd_rank']} "
		f"feature_centering={split_stats['feature_centering']} cache_in_memory={split_stats['cache_in_memory']}",
		always=True,
	)
	logger.log(f"[aug] train_runtime_enable={split_stats['train_augmentation']}", always=True)

	# --- Env mapping ---
	domain_cfg = cfg.get("domain_adaptation", {})
	use_dann = bool(domain_cfg.get("use_dann", False))
	lambda_domain = float(domain_cfg.get("lambda_domain", 0.01))
	train_envs = split_stats.get("train_envs", [])
	env_to_id = {env: idx for idx, env in enumerate(train_envs)}
	logger.log(f"[dann] use_dann={use_dann}, lambda_domain={lambda_domain}, env_to_id={env_to_id}", always=True)

	model = build_model(cfg, device, window_size=window_size, num_envs=len(env_to_id) if use_dann else 0)
	
	action_aux_cfg = cfg.get("action_aux", {})
	use_action_aux = bool(action_aux_cfg.get("enable", False))
	lambda_action_cls = float(action_aux_cfg.get("lambda_cls", 1.0))
	action_head: ActionAuxHead | None = None
	if use_action_aux:
		feature_dim = int(getattr(model, "feature_dim", 256))
		action_head = ActionAuxHead(
			input_dim=feature_dim,
			num_actions=int(action_aux_cfg.get("num_actions", len(ACTION_NAMES))),
			dropout=float(action_aux_cfg.get("dropout", 0.1)),
		).to(device)
		logger.log(f"[run] action_aux ON lambda_cls={lambda_action_cls:.2f}", always=True)

	loss_cfg = cfg.get("loss", {})
	criterion = PoseStructureLoss(
		lambda_pose=float(loss_cfg.get("lambda_pose", 1.0)),
		lambda_dist=float(loss_cfg.get("lambda_len", 0.0)),
		lambda_var=float(loss_cfg.get("lambda_var", 0.0)),
	)
	
	optimizer_name = train_cfg.get("optimizer", "AdamW")
	lr = float(train_cfg.get("lr", 3e-4))
	weight_decay = float(train_cfg.get("weight_decay", 1e-4))
	trainable_params = list(model.parameters())
	if action_head is not None: trainable_params.extend(action_head.parameters())
	
	if optimizer_name.lower() == "adamw":
		optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
	else:
		optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

	scheduler_cfg = train_cfg.get("lr_scheduler", {})
	logger.log(
		f"[sched] enable={bool(scheduler_cfg.get('enable', False))} "
		f"name={scheduler_cfg.get('name', 'cosine')} warmup_epochs={int(scheduler_cfg.get('warmup_epochs', 0))}",
		always=True,
	)
	early_stop_cfg = resolve_early_stop_config(train_cfg)
	logger.log(
		f"[early-stop] enable={early_stop_cfg['enable']} patience={early_stop_cfg['patience']} "
		f"min_epochs={early_stop_cfg['min_epochs']} min_delta={early_stop_cfg['min_delta']:.6f}",
		always=True,
	)

	checkpoint_cfg = cfg.get("checkpoint", {})
	ckpt_path = resolve_path(checkpoint_cfg.get("save_path", "checkpoints/run_ckpt.pth"), PROJECT_ROOT)
	last_ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_last{ckpt_path.suffix}")
	
	start_epoch = 0
	if args.resume and ckpt_path.exists():
		start_epoch = load_checkpoint(ckpt_path, model, optimizer, extra_modules={"action_head": action_head})
		logger.log(f"[ckpt] resumed_from={ckpt_path} start_epoch={start_epoch+1}", always=True)

	epochs = int(train_cfg.get("epochs", 50))
	best_selection_score = float("-inf")
	best_val = float("inf")
	best_epoch = start_epoch
	best_val_epoch = 0
	patience_best_val = float("inf")
	patience_best_epoch = 0
	first_degradation_epoch: int | None = None
	history_rows: list[dict[str, Any]] = []
	stopped_early = False

	for epoch in range(start_epoch, epochs):
		ep = epoch + 1
		current_lr = resolve_epoch_lr(train_cfg, epoch)
		set_optimizer_lr(optimizer, current_lr)
		logger.log(f"[lr] epoch={ep}/{epochs} lr={current_lr:.6e}", always=True)
		
		train_loss, train_stats = train_one_epoch(
			model, action_head, train_loader, device, optimizer, criterion,
			lambda_action_cls, ep, epochs, logger, log_interval=10,
			use_dann=use_dann, lambda_domain=lambda_domain, env_to_id=env_to_id, max_batches=max_train_batches
		)
		val_loss, val_nm, val_parts = evaluate(
			model, action_head, val_loader, device, criterion, lambda_action_cls, use_dann, env_to_id, max_batches=max_val_batches
		)
		selection_score, selection_mode = compute_checkpoint_selection_score(val_nm, val_parts['std_ratio'], checkpoint_cfg)
		prev_best_val = best_val
		if first_degradation_epoch is None and math.isfinite(prev_best_val) and val_nm > (prev_best_val + early_stop_cfg["min_delta"]):
			first_degradation_epoch = ep
		improved_checkpoint = selection_score > best_selection_score
		if improved_checkpoint:
			best_selection_score = selection_score
			best_epoch = ep
			best_val = val_nm
			best_val_epoch = ep
		if val_nm < (patience_best_val - early_stop_cfg["min_delta"]):
			patience_best_val = val_nm
			patience_best_epoch = ep
		val_gap_from_best = max(0.0, val_nm - best_val) if math.isfinite(best_val) else 0.0
		epochs_since_best = max(0, ep - best_val_epoch)
		epochs_since_patience_best = max(0, ep - patience_best_epoch)

		history_rows.append({
			"epoch": ep,
			"lr": current_lr,
			"train_loss": train_loss,
			"train_pose": train_stats["pose_loss"],
			"train_std_ratio": train_stats["std_ratio"],
			"train_acc": train_stats["action_acc"],
			"train_domain_loss": train_stats["domain_loss"],
			"train_domain_acc": train_stats["domain_acc"],
			"val_loss": val_loss,
			"val_nmpjpe": val_nm,
			"val_pose": val_parts["pose_loss"],
			"val_std_ratio": val_parts["std_ratio"],
			"val_acc": val_parts["action_acc"],
			"val_domain_loss": val_parts["domain_loss"],
			"val_domain_acc": val_parts["domain_acc"],
			"selection_mode": selection_mode,
			"selection_score": selection_score,
			"best_val_nmpjpe_so_far": best_val,
			"best_val_epoch_so_far": best_val_epoch,
			"epochs_since_best": epochs_since_best,
			"val_gap_from_best": val_gap_from_best,
		})
		save_history_csv(history_path, history_rows)

		logger.log(
			f"[summary] Epoch {ep}/{epochs} | "
			f"Train Loss: {train_loss:.4f} (Pose: {train_stats['pose_loss']:.4f}, StdRatio: {train_stats['std_ratio']:.3f}, Acc: {train_stats['action_acc']:.2%}, D_Acc: {train_stats.get('domain_acc', 0.0):.2%}) | "
			f"Val   Loss: {val_loss:.4f} (nMPJPE: {val_nm:.4f}, StdRatio: {val_parts['std_ratio']:.3f}, Acc: {val_parts['action_acc']:.2%}, D_Acc: {val_parts.get('domain_acc', 0.0):.2%}) | "
			f"Score: {selection_score:.4f} | BestVal: {best_val:.4f}@{best_val_epoch} | Gap: {val_gap_from_best:.4f}",
			always=True
		)

		if ep > 0:
			save_checkpoint(last_ckpt_path, model, optimizer, ep, cfg, {"action_head": action_head})
		
		if improved_checkpoint:
			save_checkpoint(ckpt_path, model, optimizer, ep, cfg, {"action_head": action_head})
			logger.log(f"[ckpt] New best model saved (Score: {best_selection_score:.4f}, nMPJPE: {val_nm:.4f})", always=True)

		if early_stop_cfg["enable"] and ep >= early_stop_cfg["min_epochs"] and epochs_since_patience_best >= early_stop_cfg["patience"]:
			logger.log(
				f"[early-stop] Triggered at epoch {ep}. patience_best_epoch={patience_best_epoch} "
				f"patience_best_val={patience_best_val:.4f} checkpoint_best_epoch={best_val_epoch} "
				f"checkpoint_best_val={best_val:.4f} patience={early_stop_cfg['patience']}",
				always=True,
			)
			stopped_early = True
			break

	load_checkpoint(ckpt_path, model, extra_modules={"action_head": action_head})
	train_loss, train_nm, train_parts = evaluate(
		model, action_head, train_loader, device, criterion, lambda_action_cls, use_dann, env_to_id, max_batches=max_train_eval_batches
	)
	test_loss, test_nm, test_parts = evaluate(
		model, action_head, test_loader, device, criterion, lambda_action_cls, use_dann, env_to_id, max_batches=max_test_batches
	)
	
	logger.log(f"[train eval] nMPJPE={train_nm:.4f}", always=True)
	logger.log(f"[test eval] nMPJPE={test_nm:.4f} (Acc: {test_parts['action_acc']:.2%})", always=True)
	overfit_summary = summarize_overfit_history(history_rows)
	logger.log(
		f"[overfit] best_val_epoch={overfit_summary['best_val_epoch']} "
		f"first_degradation_epoch={overfit_summary['first_degradation_epoch']} "
		f"final_val_gap={float(overfit_summary['final_val_gap']):.4f} "
		f"epochs_since_best={overfit_summary['epochs_since_best']} stopped_early={stopped_early}",
		always=True,
	)
	
	assessment, reason = assess_fit(history_rows, test_nm)
	logger.log(f"[assessment] {assessment}: {reason}", always=True)

	if skip_artifact_prune:
		logger.log("[prune] Skipped artifact pruning for debug/smoke run.", always=True)
	else:
		try:
			prune_cmd = [sys.executable, str(PROJECT_ROOT / "tools" / "prune_run_artifacts.py"), "--keep", "5"]
			subprocess.run(prune_cmd, check=True)
			logger.log("[prune] Kept the latest 5 artifacts.", always=True)
		except Exception as e:
			logger.log(f"[prune] Warn: Failed to prune {e}", always=True)

if __name__ == "__main__":
	main()
