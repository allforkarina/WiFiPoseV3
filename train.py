from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

import yaml

from dataloader.aoa_dataset import AOASampleDataset, sample_to_env
from dataloader.stratified_sampler import StratifiedBatchSampler
from mymodels import ConvBaseline, MultiScaleTemporalPoseTCN, ResNet1DPose
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
		lambda_rel: float = 1.0,
		lambda_dist: float = 0.5,
		lambda_dir: float = 0.2,
		lambda_var: float = 0.1,
		lambda_batch_div: float = 0.0,
		lambda_inter_div: float = 0.0,
		lambda_intra_div: float = 0.0,
		preserve_ratio: float = 0.6,
		huber_beta: float = 0.05,
	) -> None:
		super().__init__()
		self.lambda_pose = float(lambda_pose)
		self.lambda_rel = float(lambda_rel)
		self.lambda_dist = float(lambda_dist)
		self.lambda_dir = float(lambda_dir)
		self.lambda_var = float(lambda_var)
		self.lambda_batch_div = float(lambda_batch_div)
		self.lambda_inter_div = float(lambda_inter_div)
		self.lambda_intra_div = float(lambda_intra_div)
		self.preserve_ratio = float(preserve_ratio)
		self.huber_beta = float(huber_beta)

	def _pair_diversity_terms(
		self,
		pred_sample_dist: torch.Tensor,
		target_sample_dist: torch.Tensor,
		mask: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		if bool(mask.any()):
			pred_pairs = pred_sample_dist[mask]
			target_pairs = target_sample_dist[mask]
			target_pair_mean = target_pairs.mean().clamp(min=1e-6)
			div_loss = F.smooth_l1_loss(
				pred_pairs / target_pair_mean,
				target_pairs / target_pair_mean,
				beta=self.huber_beta,
			)
			pair_ratio = pred_pairs.mean() / target_pairs.mean().clamp(min=1e-8)
			return div_loss, pred_pairs.mean(), target_pairs.mean(), pair_ratio

		zero = pred_sample_dist.new_zeros(())
		return zero, zero, zero, zero

	def forward(
		self,
		pred: torch.Tensor,
		target: torch.Tensor,
		action_targets: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
		pose_loss = F.smooth_l1_loss(pred, target, beta=self.huber_beta)

		pred_rel = pred.unsqueeze(2) - pred.unsqueeze(1)
		target_rel = target.unsqueeze(2) - target.unsqueeze(1)
		rel_loss = F.smooth_l1_loss(pred_rel, target_rel, beta=self.huber_beta)

		pred_dist = torch.cdist(pred, pred, p=2)
		target_dist = torch.cdist(target, target, p=2)
		dist_loss = F.l1_loss(pred_dist, target_dist)

		pred_centered = pred - pred.mean(dim=1, keepdim=True)
		target_centered = target - target.mean(dim=1, keepdim=True)
		pred_unit = F.normalize(pred_centered, dim=-1, eps=1e-6)
		target_unit = F.normalize(target_centered, dim=-1, eps=1e-6)
		dir_loss = (1.0 - (pred_unit * target_unit).sum(dim=-1)).mean()

		pred_flat = pred.view(pred.size(0), -1)
		target_flat = target.view(target.size(0), -1)
		pred_std = pred_flat.std(dim=0, unbiased=False)
		target_std = target_flat.std(dim=0, unbiased=False)
		var_loss = F.relu((target_std * self.preserve_ratio) - pred_std).mean()
		std_ratio = pred_std.mean() / target_std.mean().clamp(min=1e-8)

		if pred.size(0) >= 2:
			pred_sample_dist = torch.cdist(pred_flat, pred_flat, p=2)
			target_sample_dist = torch.cdist(target_flat, target_flat, p=2)
			off_diag = ~torch.eye(pred.size(0), dtype=torch.bool, device=pred.device)
			batch_div_loss, pred_pair_mean, target_pair_mean, pair_dist_ratio = self._pair_diversity_terms(
				pred_sample_dist,
				target_sample_dist,
				off_diag,
			)
			if action_targets is not None and action_targets.numel() == pred.size(0):
				same_action = action_targets.unsqueeze(0) == action_targets.unsqueeze(1)
				inter_mask = off_diag & (~same_action)
				intra_mask = off_diag & same_action
				inter_div_loss, pred_inter_mean, target_inter_mean, inter_pair_ratio = self._pair_diversity_terms(
					pred_sample_dist,
					target_sample_dist,
					inter_mask,
				)
				intra_div_loss, pred_intra_mean, target_intra_mean, intra_pair_ratio = self._pair_diversity_terms(
					pred_sample_dist,
					target_sample_dist,
					intra_mask,
				)
			else:
				inter_div_loss = pred_flat.new_zeros(())
				intra_div_loss = pred_flat.new_zeros(())
				pred_inter_mean = pred_flat.new_zeros(())
				target_inter_mean = pred_flat.new_zeros(())
				inter_pair_ratio = pred_flat.new_zeros(())
				pred_intra_mean = pred_flat.new_zeros(())
				target_intra_mean = pred_flat.new_zeros(())
				intra_pair_ratio = pred_flat.new_zeros(())
		else:
			batch_div_loss = pred_flat.new_zeros(())
			pred_pair_mean = pred_flat.new_zeros(())
			target_pair_mean = pred_flat.new_zeros(())
			pair_dist_ratio = pred_flat.new_zeros(())
			inter_div_loss = pred_flat.new_zeros(())
			intra_div_loss = pred_flat.new_zeros(())
			pred_inter_mean = pred_flat.new_zeros(())
			target_inter_mean = pred_flat.new_zeros(())
			inter_pair_ratio = pred_flat.new_zeros(())
			pred_intra_mean = pred_flat.new_zeros(())
			target_intra_mean = pred_flat.new_zeros(())
			intra_pair_ratio = pred_flat.new_zeros(())

		total = (
			self.lambda_pose * pose_loss
			+ self.lambda_rel * rel_loss
			+ self.lambda_dist * dist_loss
			+ self.lambda_dir * dir_loss
			+ self.lambda_var * var_loss
			+ self.lambda_batch_div * batch_div_loss
			+ self.lambda_inter_div * inter_div_loss
			+ self.lambda_intra_div * intra_div_loss
		)
		parts = {
			"pose_loss": pose_loss.detach(),
			"rel_loss": rel_loss.detach(),
			"dist_loss": dist_loss.detach(),
			"dir_loss": dir_loss.detach(),
			"var_loss": var_loss.detach(),
			"batch_div_loss": batch_div_loss.detach(),
			"inter_div_loss": inter_div_loss.detach(),
			"intra_div_loss": intra_div_loss.detach(),
			"pred_std_mean": pred_std.mean().detach(),
			"target_std_mean": target_std.mean().detach(),
			"std_ratio": std_ratio.detach(),
			"pred_pair_dist_mean": pred_pair_mean.detach(),
			"target_pair_dist_mean": target_pair_mean.detach(),
			"pair_dist_ratio": pair_dist_ratio.detach(),
			"pred_inter_pair_dist_mean": pred_inter_mean.detach(),
			"target_inter_pair_dist_mean": target_inter_mean.detach(),
			"inter_pair_dist_ratio": inter_pair_ratio.detach(),
			"pred_intra_pair_dist_mean": pred_intra_mean.detach(),
			"target_intra_pair_dist_mean": target_intra_mean.detach(),
			"intra_pair_dist_ratio": intra_pair_ratio.detach(),
			"total_loss": total.detach(),
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


def resolve_data_roots(cfg: Dict[str, Any], aoa_override: str | None, labels_override: str | None) -> tuple[Path, Path]:
	data_root = resolve_path(cfg.get("data_root"), PROJECT_ROOT / "data")
	aoa_root = resolve_path(aoa_override or cfg.get("aoa_cache_root"), data_root / "aoa_cache")
	labels_root = resolve_path(labels_override or cfg.get("labels_root"), data_root / "dataset")
	return aoa_root, labels_root


def resolve_normalize_mode(cfg: Dict[str, Any]) -> str:
	loss_cfg = cfg.get("loss", {})
	return str(loss_cfg.get("normalize_mode", "pelvis_torso")).strip().lower() or "pelvis_torso"


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
	loss_cfg = cfg.setdefault("loss", {})
	checkpoint_cfg = cfg.setdefault("checkpoint", {})
	selection_cfg = checkpoint_cfg.setdefault("selection", {})
	logging_cfg = cfg.setdefault("logging", {})
	action_aux_cfg = cfg.setdefault("action_aux", {})

	if args.normalize_mode is not None:
		loss_cfg["normalize_mode"] = args.normalize_mode
	if args.lambda_var is not None:
		loss_cfg["lambda_var"] = float(args.lambda_var)
	if args.lambda_batch_div is not None:
		loss_cfg["lambda_batch_div"] = float(args.lambda_batch_div)
	if args.lambda_inter_div is not None:
		loss_cfg["lambda_inter_div"] = float(args.lambda_inter_div)
	if args.lambda_intra_div is not None:
		loss_cfg["lambda_intra_div"] = float(args.lambda_intra_div)
	if bool(args.disable_batch_div_schedule):
		loss_cfg["batch_div_schedule"] = {"enable": False}
	if bool(args.disable_inter_div_schedule):
		loss_cfg["inter_div_schedule"] = {"enable": False}
	if bool(args.disable_intra_div_schedule):
		loss_cfg["intra_div_schedule"] = {"enable": False}
	if args.selection_mode is not None:
		selection_cfg["mode"] = args.selection_mode
	if bool(args.enable_action_aux):
		action_aux_cfg["enable"] = True
	if args.lambda_action_cls is not None:
		action_aux_cfg["lambda_cls"] = float(args.lambda_action_cls)
	if args.action_aux_dropout is not None:
		action_aux_cfg["dropout"] = float(args.action_aux_dropout)
	if args.experiment_name:
		logging_cfg["run_name"] = args.experiment_name


def resolve_scheduled_lambda(
	loss_cfg: Dict[str, Any],
	base_key: str,
	schedule_key: str,
	epoch: int,
	num_epochs: int,
) -> float:
	base_value = float(loss_cfg.get(base_key, 0.0))
	schedule_cfg = loss_cfg.get(schedule_key, {})
	if not isinstance(schedule_cfg, dict) or not bool(schedule_cfg.get("enable", False)):
		return base_value

	name = str(schedule_cfg.get("name", "fixed")).lower()
	if num_epochs <= 1 or name == "fixed":
		return float(schedule_cfg.get("start", base_value))

	start = float(schedule_cfg.get("start", base_value))
	peak = float(schedule_cfg.get("peak", base_value))
	end = float(schedule_cfg.get("end", peak))
	progress = 0.0 if num_epochs <= 1 else (epoch - 1) / max(1, num_epochs - 1)

	if name == "linear":
		return start + (end - start) * progress

	if name == "rise_fall":
		peak_ratio = float(schedule_cfg.get("peak_epoch_ratio", 0.4))
		peak_ratio = min(max(peak_ratio, 1e-6), 1.0)
		if progress <= peak_ratio:
			local_progress = progress / peak_ratio
			return start + (peak - start) * local_progress
		local_progress = (progress - peak_ratio) / max(1e-6, 1.0 - peak_ratio)
		return peak + (end - peak) * local_progress

	return base_value


def compute_checkpoint_selection_score(
	val_nmpjpe: float,
	val_parts: dict[str, float],
	checkpoint_cfg: dict[str, Any],
) -> tuple[float, str]:
	selection_cfg = checkpoint_cfg.get("selection", {})
	mode = str(selection_cfg.get("mode", "accuracy")).lower()
	if mode == "accuracy":
		return -float(val_nmpjpe), "accuracy"

	if mode == "diversity_first":
		pair_w = float(selection_cfg.get("pair_ratio_weight", 1.0))
		std_w = float(selection_cfg.get("std_ratio_weight", 0.35))
		inter_w = float(selection_cfg.get("inter_pair_ratio_weight", 0.15))
		nm_penalty = float(selection_cfg.get("nmpjpe_penalty", 0.02))
		score = (
			pair_w * float(val_parts.get("pair_dist_ratio", 0.0))
			+ std_w * float(val_parts.get("std_ratio", 0.0))
			+ inter_w * float(val_parts.get("inter_pair_dist_ratio", 0.0))
			- nm_penalty * float(val_nmpjpe)
		)
		return score, "diversity_first"

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


def forward_pose_with_features(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	if not hasattr(model, "forward_features") or not hasattr(model, "forward_head"):
		raise AttributeError("Model must implement forward_features() and forward_head() for anti-collapse training.")
	features = model.forward_features(x)
	pred = model.forward_head(features)
	return pred, features


def split_indices_by_envs(
        ds: AOASampleDataset,
        val_env: str,
        test_env: str,
) -> tuple[list[int], list[int], list[int]]:
        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []

        import sys
        train_env_str = ""
        if '--train_env' in sys.argv:
                train_env_str = sys.argv[sys.argv.index('--train_env') + 1]

        train_envs = [e.strip() for e in train_env_str.split(',')] if train_env_str else []
        val_envs = [e.strip() for e in val_env.split(',')]
        test_envs = [e.strip() for e in test_env.split(',')]

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


def build_subset_loader(
        ds: AOASampleDataset,
        indices: list[int],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        shuffle: bool,
        use_stratified: bool,
) -> DataLoader:
	subset = Subset(ds, indices)
	if use_stratified:
		meta = [(ds.index[idx][0], sample_to_env(ds.index[idx][1])) for idx in indices]
		sampler = StratifiedBatchSampler(meta=meta, batch_size=batch_size, shuffle=shuffle, drop_last=False)
		return DataLoader(subset, batch_sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
	return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


def build_dataloaders(
	cfg: Dict[str, Any],
	args: argparse.Namespace,
	device: torch.device,
	aoa_root: Path,
	labels_root: Path,
	val_env: str,
	test_env: str,
	window_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
	ds = AOASampleDataset(
		aoa_root=aoa_root,
		labels_root=labels_root,
		window_size=window_size,
		normalize_mode=resolve_normalize_mode(cfg),
	)
	if len(ds) == 0:
		raise RuntimeError(f"Dataset is empty. aoa_root={aoa_root} labels_root={labels_root}")

	batch_size = int(cfg.get("dataset", {}).get("batch_size", 32))
	num_workers = int(cfg.get("dataset", {}).get("num_workers", 0))
	pin_memory = bool(cfg.get("dataset", {}).get("pin_memory", False)) and device.type == "cuda"

	train_indices, val_indices, test_indices = split_indices_by_envs(ds, val_env=val_env, test_env=test_env)
	if not train_indices:
		raise RuntimeError(f"Training split is empty for val_env={val_env} test_env={test_env}")
	if not val_indices:
		raise RuntimeError(f"Validation split is empty for val_env={val_env}")
	if not test_indices:
		raise RuntimeError(f"Test split is empty for test_env={test_env}")

	train_loader = build_subset_loader(ds, train_indices, batch_size, num_workers, pin_memory, not args.single_batch, args.use_stratified and not args.single_batch)
	if args.single_batch:
		# Keep only the very first batch
		# Subset of subset - we take just `batch_size` items from `train_indices`
		single_batch_indices = train_indices[:batch_size]
		train_loader = build_subset_loader(ds, single_batch_indices, batch_size, num_workers, pin_memory, False, False)
		val_loader = build_subset_loader(ds, val_indices[:batch_size], batch_size, num_workers, pin_memory, False, False)
		test_loader = build_subset_loader(ds, test_indices[:batch_size], batch_size, num_workers, pin_memory, False, False)
	else:
		val_loader = build_subset_loader(ds, val_indices, batch_size, num_workers, pin_memory, False, False)
		test_loader = build_subset_loader(ds, test_indices, batch_size, num_workers, pin_memory, False, False)
	
	stats = {
		"dataset_size": len(ds),
		"train_size": len(train_indices),
		"val_size": len(val_indices),
		"test_size": len(test_indices),
		"batch_size": batch_size,
		"window_size": window_size,
		"val_env": val_env,
		"test_env": test_env,
	}
	return train_loader, val_loader, test_loader, stats


def build_model(cfg: Dict[str, Any], device: torch.device, model_name_override: str | None, window_size: int | None = None, num_envs: int = 0) -> nn.Module:
	mcfg = cfg.get("model", {})
	model_name = (model_name_override or mcfg.get("name", "conv1d_baseline")).lower()
	effective_window = int(window_size if window_size is not None else cfg.get("dataset", {}).get("window_size", 1))
	common_kwargs = {
		"input_channels": effective_window if model_name in {"conv1d_baseline", "resnet1d", "ms_tcn_pose"} else mcfg.get("input_channels", 1),
		"input_length": mcfg.get("input_length", 181),
		"hidden_dim": mcfg.get("hidden_dim", 256),
		"num_joints": mcfg.get("output_joints", 17),
		"out_dim": mcfg.get("output_dim", 2),
		"dropout": mcfg.get("dropout", 0.2),
	}
	if model_name == "conv1d_baseline":
		model = ConvBaseline(**common_kwargs)
	elif model_name == "resnet1d":
		model = ResNet1DPose(**common_kwargs)
	elif model_name == "ms_tcn_pose":
		model = MultiScaleTemporalPoseTCN(**common_kwargs)
	else:
		raise ValueError(f"Unsupported model name: {model_name}")
	return model.to(device)


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
		state["extra_modules"] = {
			name: module.state_dict()
			for name, module in extra_modules.items()
			if module is not None
		}
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
	max_steps: int,
	epoch: int,
	num_epochs: int,
	logger: TrainLogger,
	log_interval: int,
) -> tuple[float, Dict[str, Any]]:
	model.train()
	if action_head is not None:
		action_head.train()
	running_loss = 0.0
	running_pose_loss = 0.0
	running_rel_loss = 0.0
	running_dist_loss = 0.0
	running_dir_loss = 0.0
	running_var_loss = 0.0
	running_batch_div_loss = 0.0
	running_inter_div_loss = 0.0
	running_intra_div_loss = 0.0
	running_pred_std = 0.0
	running_target_std = 0.0
	running_std_ratio = 0.0
	running_pred_pair_dist = 0.0
	running_target_pair_dist = 0.0
	running_pair_dist_ratio = 0.0
	running_pred_inter_pair_dist = 0.0
	running_target_inter_pair_dist = 0.0
	running_inter_pair_dist_ratio = 0.0
	running_pred_intra_pair_dist = 0.0
	running_target_intra_pair_dist = 0.0
	running_intra_pair_dist_ratio = 0.0
	running_action_cls_loss = 0.0
	running_action_acc = 0.0
	grad_norms: list[float] = []
	has_nan_loss = False
	has_nan_grad = False
	actions_seen: set[str] = set()
	envs_seen: set[str] = set()

	total_steps = min(max_steps, len(loader)) if max_steps > 0 else len(loader)
	for step, (x, y, meta) in enumerate(loader):
		x = x.to(device)
		y = y.to(device)
		action_targets = build_action_targets(meta, device)
		optimizer.zero_grad()
		pred, features = forward_pose_with_features(model, x)
		loss, loss_parts = criterion(pred, y, action_targets=action_targets)
		if action_head is not None:
			action_logits = action_head(features)
			action_cls_loss = F.cross_entropy(action_logits, action_targets)
			action_acc = (action_logits.argmax(dim=1) == action_targets).float().mean()
			loss = loss + (lambda_action_cls * action_cls_loss)
		else:
			action_cls_loss = loss.detach().new_zeros(())
			action_acc = loss.detach().new_zeros(())
		if not torch.isfinite(loss):
			logger.log(f"[warn] epoch={epoch} step={step+1} non-finite loss: {loss.item()}", always=True)
			has_nan_loss = True
			break
		loss.backward()
		grad_norm = _compute_grad_norm(model, action_head)
		if not math.isfinite(grad_norm):
			has_nan_grad = True
		log_msg = f"[train] epoch={epoch}/{num_epochs} step={step+1}/{total_steps} loss={loss.item():.6f} grad_norm={grad_norm:.3f}"
		if action_head is not None:
			log_msg += f" action_cls={action_cls_loss.item():.6f} action_acc={action_acc.item():.3f}"
		logger.log(log_msg, always=((step + 1) % log_interval == 0 or step == 0))
		optimizer.step()

		grad_norms.append(grad_norm)
		running_loss += loss.item()
		running_pose_loss += float(loss_parts["pose_loss"].item())
		running_rel_loss += float(loss_parts["rel_loss"].item())
		running_dist_loss += float(loss_parts["dist_loss"].item())
		running_dir_loss += float(loss_parts["dir_loss"].item())
		running_var_loss += float(loss_parts["var_loss"].item())
		running_batch_div_loss += float(loss_parts["batch_div_loss"].item())
		running_inter_div_loss += float(loss_parts["inter_div_loss"].item())
		running_intra_div_loss += float(loss_parts["intra_div_loss"].item())
		running_pred_std += float(loss_parts["pred_std_mean"].item())
		running_target_std += float(loss_parts["target_std_mean"].item())
		running_std_ratio += float(loss_parts["std_ratio"].item())
		running_pred_pair_dist += float(loss_parts["pred_pair_dist_mean"].item())
		running_target_pair_dist += float(loss_parts["target_pair_dist_mean"].item())
		running_pair_dist_ratio += float(loss_parts["pair_dist_ratio"].item())
		running_pred_inter_pair_dist += float(loss_parts["pred_inter_pair_dist_mean"].item())
		running_target_inter_pair_dist += float(loss_parts["target_inter_pair_dist_mean"].item())
		running_inter_pair_dist_ratio += float(loss_parts["inter_pair_dist_ratio"].item())
		running_pred_intra_pair_dist += float(loss_parts["pred_intra_pair_dist_mean"].item())
		running_target_intra_pair_dist += float(loss_parts["target_intra_pair_dist_mean"].item())
		running_intra_pair_dist_ratio += float(loss_parts["intra_pair_dist_ratio"].item())
		running_action_cls_loss += float(action_cls_loss.item())
		running_action_acc += float(action_acc.item())
		actions_seen.update(extract_meta_field(meta, "action"))
		envs_seen.update(extract_meta_field(meta, "env_id"))
		if step + 1 >= total_steps:
			break

	avg_loss = running_loss / max(1, total_steps)
	stats: Dict[str, Any] = {
		"grad_norm_mean": float(sum(grad_norms) / max(1, len(grad_norms))),
		"has_nan_loss": has_nan_loss,
		"has_nan_grad": has_nan_grad,
		"pose_loss_mean": running_pose_loss / max(1, total_steps),
		"rel_loss_mean": running_rel_loss / max(1, total_steps),
		"dist_loss_mean": running_dist_loss / max(1, total_steps),
		"dir_loss_mean": running_dir_loss / max(1, total_steps),
		"var_loss_mean": running_var_loss / max(1, total_steps),
		"batch_div_loss_mean": running_batch_div_loss / max(1, total_steps),
		"inter_div_loss_mean": running_inter_div_loss / max(1, total_steps),
		"intra_div_loss_mean": running_intra_div_loss / max(1, total_steps),
		"pred_std_mean": running_pred_std / max(1, total_steps),
		"target_std_mean": running_target_std / max(1, total_steps),
		"std_ratio_mean": running_std_ratio / max(1, total_steps),
		"pred_pair_dist_mean": running_pred_pair_dist / max(1, total_steps),
		"target_pair_dist_mean": running_target_pair_dist / max(1, total_steps),
		"pair_dist_ratio_mean": running_pair_dist_ratio / max(1, total_steps),
		"pred_inter_pair_dist_mean": running_pred_inter_pair_dist / max(1, total_steps),
		"target_inter_pair_dist_mean": running_target_inter_pair_dist / max(1, total_steps),
		"inter_pair_dist_ratio_mean": running_inter_pair_dist_ratio / max(1, total_steps),
		"pred_intra_pair_dist_mean": running_pred_intra_pair_dist / max(1, total_steps),
		"target_intra_pair_dist_mean": running_target_intra_pair_dist / max(1, total_steps),
		"intra_pair_dist_ratio_mean": running_intra_pair_dist_ratio / max(1, total_steps),
		"action_cls_loss_mean": running_action_cls_loss / max(1, total_steps),
		"action_acc_mean": running_action_acc / max(1, total_steps),
		"actions_covered": len([x for x in actions_seen if x]),
		"envs_covered": len([x for x in envs_seen if x]),
	}
	return avg_loss, stats


def evaluate(
	model: nn.Module,
	action_head: ActionAuxHead | None,
	loader: DataLoader,
	device: torch.device,
	criterion: PoseStructureLoss,
	lambda_action_cls: float,
) -> tuple[float, float, dict[str, float]]:
	model.eval()
	if action_head is not None:
		action_head.eval()
	total_loss = 0.0
	total_err = 0.0
	total_cnt = 0
	total_pose = 0.0
	total_rel = 0.0
	total_dist = 0.0
	total_dir = 0.0
	total_var = 0.0
	total_batch_div = 0.0
	total_inter_div = 0.0
	total_intra_div = 0.0
	total_pred_std = 0.0
	total_target_std = 0.0
	total_std_ratio = 0.0
	total_pred_pair_dist = 0.0
	total_target_pair_dist = 0.0
	total_pair_dist_ratio = 0.0
	total_pred_inter_pair_dist = 0.0
	total_target_inter_pair_dist = 0.0
	total_inter_pair_dist_ratio = 0.0
	total_pred_intra_pair_dist = 0.0
	total_target_intra_pair_dist = 0.0
	total_intra_pair_dist_ratio = 0.0
	total_action_cls = 0.0
	total_action_acc = 0.0
	with torch.no_grad():
		for x, y, meta in loader:
			x = x.to(device)
			y = y.to(device)
			action_targets = build_action_targets(meta, device)
			pred, features = forward_pose_with_features(model, x)
			loss, loss_parts = criterion(pred, y, action_targets=action_targets)
			if action_head is not None:
				action_logits = action_head(features)
				action_cls_loss = F.cross_entropy(action_logits, action_targets)
				action_acc = (action_logits.argmax(dim=1) == action_targets).float().mean()
				loss = loss + (lambda_action_cls * action_cls_loss)
			else:
				action_cls_loss = loss.detach().new_zeros(())
				action_acc = loss.detach().new_zeros(())
			err = nMPJPE(pred, y)
			batch_size = x.size(0)
			total_loss += loss.item() * batch_size
			total_pose += float(loss_parts["pose_loss"].item()) * batch_size
			total_rel += float(loss_parts["rel_loss"].item()) * batch_size
			total_dist += float(loss_parts["dist_loss"].item()) * batch_size
			total_dir += float(loss_parts["dir_loss"].item()) * batch_size
			total_var += float(loss_parts["var_loss"].item()) * batch_size
			total_batch_div += float(loss_parts["batch_div_loss"].item()) * batch_size
			total_inter_div += float(loss_parts["inter_div_loss"].item()) * batch_size
			total_intra_div += float(loss_parts["intra_div_loss"].item()) * batch_size
			total_pred_std += float(loss_parts["pred_std_mean"].item()) * batch_size
			total_target_std += float(loss_parts["target_std_mean"].item()) * batch_size
			total_std_ratio += float(loss_parts["std_ratio"].item()) * batch_size
			total_pred_pair_dist += float(loss_parts["pred_pair_dist_mean"].item()) * batch_size
			total_target_pair_dist += float(loss_parts["target_pair_dist_mean"].item()) * batch_size
			total_pair_dist_ratio += float(loss_parts["pair_dist_ratio"].item()) * batch_size
			total_pred_inter_pair_dist += float(loss_parts["pred_inter_pair_dist_mean"].item()) * batch_size
			total_target_inter_pair_dist += float(loss_parts["target_inter_pair_dist_mean"].item()) * batch_size
			total_inter_pair_dist_ratio += float(loss_parts["inter_pair_dist_ratio"].item()) * batch_size
			total_pred_intra_pair_dist += float(loss_parts["pred_intra_pair_dist_mean"].item()) * batch_size
			total_target_intra_pair_dist += float(loss_parts["target_intra_pair_dist_mean"].item()) * batch_size
			total_intra_pair_dist_ratio += float(loss_parts["intra_pair_dist_ratio"].item()) * batch_size
			total_action_cls += float(action_cls_loss.item()) * batch_size
			total_action_acc += float(action_acc.item()) * batch_size
			total_err += err.item() * batch_size
			total_cnt += batch_size
	parts = {
		"pose_loss": total_pose / max(1, total_cnt),
		"rel_loss": total_rel / max(1, total_cnt),
		"dist_loss": total_dist / max(1, total_cnt),
		"dir_loss": total_dir / max(1, total_cnt),
		"var_loss": total_var / max(1, total_cnt),
		"batch_div_loss": total_batch_div / max(1, total_cnt),
		"inter_div_loss": total_inter_div / max(1, total_cnt),
		"intra_div_loss": total_intra_div / max(1, total_cnt),
		"pred_std_mean": total_pred_std / max(1, total_cnt),
		"target_std_mean": total_target_std / max(1, total_cnt),
		"std_ratio": total_std_ratio / max(1, total_cnt),
		"pred_pair_dist_mean": total_pred_pair_dist / max(1, total_cnt),
		"target_pair_dist_mean": total_target_pair_dist / max(1, total_cnt),
		"pair_dist_ratio": total_pair_dist_ratio / max(1, total_cnt),
		"pred_inter_pair_dist_mean": total_pred_inter_pair_dist / max(1, total_cnt),
		"target_inter_pair_dist_mean": total_target_inter_pair_dist / max(1, total_cnt),
		"inter_pair_dist_ratio": total_inter_pair_dist_ratio / max(1, total_cnt),
		"pred_intra_pair_dist_mean": total_pred_intra_pair_dist / max(1, total_cnt),
		"target_intra_pair_dist_mean": total_target_intra_pair_dist / max(1, total_cnt),
		"intra_pair_dist_ratio": total_intra_pair_dist_ratio / max(1, total_cnt),
		"action_cls_loss": total_action_cls / max(1, total_cnt),
		"action_acc": total_action_acc / max(1, total_cnt),
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


def plot_history(path: Path, rows: list[dict[str, Any]], logger: TrainLogger) -> None:
	if not rows:
		return
	try:
		import matplotlib.pyplot as plt
	except ImportError:
		logger.log("[plot] matplotlib not installed; skipped loss/nMPJPE plot", always=True)
		return

	epochs = [row["epoch"] for row in rows]
	train_loss = [row["train_loss"] for row in rows]
	val_loss = [row["val_loss"] for row in rows]
	val_nmpjpe = [row["val_nmpjpe"] for row in rows]
	fig, ax1 = plt.subplots(figsize=(9, 5))
	ax1.plot(epochs, train_loss, label="train_loss", color="tab:blue", linewidth=2)
	ax1.plot(epochs, val_loss, label="val_loss", color="tab:orange", linewidth=2)
	ax1.set_xlabel("Epoch")
	ax1.set_ylabel("Loss")
	ax1.grid(True, linestyle="--", alpha=0.4)

	ax2 = ax1.twinx()
	ax2.plot(epochs, val_nmpjpe, label="val_nMPJPE", color="tab:green", linewidth=2)
	ax2.set_ylabel("nMPJPE")
	lines = ax1.get_lines() + ax2.get_lines()
	ax1.legend(lines, [line.get_label() for line in lines], loc="upper right")
	fig.tight_layout()
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(path, dpi=150)
	plt.close(fig)
	logger.log(f"[plot] saved path={path}", always=True)


def assess_fit(history_rows: list[dict[str, Any]], final_test_nmpjpe: float) -> tuple[str, str]:
	if not history_rows:
		return "unknown", "No training history available."
	best_val = min(row["val_nmpjpe"] for row in history_rows)
	final_val = history_rows[-1]["val_nmpjpe"]
	final_train = history_rows[-1]["train_loss"]
	final_val_loss = history_rows[-1]["val_loss"]
	gap = final_val_loss - final_train
	if final_val > best_val * 1.15 and gap > 0.03:
		return "overfit", (
			f"Validation nMPJPE rebounded from best {best_val:.4f} to {final_val:.4f}, "
			f"and val_loss exceeds train_loss by {gap:.4f}."
		)
	if final_train > 0.08 and final_val > 0.30:
		return "underfit", (
			f"Train loss {final_train:.4f} and val nMPJPE {final_val:.4f} are both still high, "
			f"so capacity or training budget is likely insufficient."
		)
	return "usable", (
		f"Validation stayed close to the best point ({best_val:.4f}) and final test nMPJPE is {final_test_nmpjpe:.4f}."
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Training loop for AoA pose models with train/val/test splits")
	parser.add_argument("--config", type=str, default="configs/default.yaml")
	parser.add_argument("--use_stratified", action="store_true")
	parser.add_argument("--single_batch", action="store_true", help="Extreme overfit mode: train on a single batch iteratively")
	parser.add_argument("--max_steps", type=int, default=0, help="Max training steps per epoch (0 = full epoch)")
	parser.add_argument("--checkpoint", type=str, default="checkpoints/debug_ckpt.pth")
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--verbose", action="store_true")
	parser.add_argument("--log_interval", type=int, default=10)
	parser.add_argument("--aoa_cache_root", type=str, default=None)
	parser.add_argument("--labels_root", type=str, default=None)
	parser.add_argument("--val_env", type=str, default="env3")
	parser.add_argument("--test_env", type=str, default="env4")
	parser.add_argument("--train_env", type=str, default=None)
	parser.add_argument("--epochs", type=int, default=None)
	parser.add_argument("--model_name", type=str, default=None, help="conv1d_baseline | resnet1d | ms_tcn_pose")
	parser.add_argument("--window_size", type=int, default=None, help="Temporal AoA window size; even values are rounded up")
	parser.add_argument("--normalize_mode", type=str, choices=["pelvis_torso", "mean_rms"], default=None)
	parser.add_argument("--lambda_var", type=float, default=None)
	parser.add_argument("--lambda_batch_div", type=float, default=None)
	parser.add_argument("--lambda_inter_div", type=float, default=None)
	parser.add_argument("--lambda_intra_div", type=float, default=None)
	parser.add_argument("--selection_mode", type=str, choices=["accuracy", "diversity_first"], default=None)
	parser.add_argument("--enable_action_aux", action="store_true")
	parser.add_argument("--lambda_action_cls", type=float, default=None)
	parser.add_argument("--action_aux_dropout", type=float, default=None)
	parser.add_argument("--disable_batch_div_schedule", action="store_true")
	parser.add_argument("--disable_inter_div_schedule", action="store_true")
	parser.add_argument("--disable_intra_div_schedule", action="store_true")
	parser.add_argument("--experiment_name", type=str, default=None)
	args = parser.parse_args()

	config_path = Path(args.config)
	if not config_path.is_absolute():
		config_path = PROJECT_ROOT / config_path
	cfg = load_config(config_path)
	apply_cli_overrides(cfg, args)
	train_cfg = cfg.get("train", {})
	logging_cfg = cfg.get("logging", {})
	seed = int(train_cfg.get("seed", 42))
	set_seed(seed)
	window_size = int(args.window_size if args.window_size is not None else cfg.get("dataset", {}).get("window_size", 1))
	if window_size % 2 == 0:
		window_size += 1

	device_str = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device(device_str)
	aoa_root, labels_root = resolve_data_roots(cfg, args.aoa_cache_root, args.labels_root)

	logs_root = resolve_path(cfg.get("logs_root"), PROJECT_ROOT / "logs")
	log_dir = logs_root / "train"
	run_name = logging_cfg.get("run_name", "default")
	model_name = (args.model_name or cfg.get("model", {}).get("name", "conv1d_baseline")).lower()
	cfg.setdefault("dataset", {})
	cfg.setdefault("model", {})
	cfg["dataset"]["window_size"] = window_size
	cfg["model"]["name"] = model_name
	ts = datetime.now().strftime("%Y%m%d-%H%M%S")
	log_path = log_dir / f"{run_name}_{model_name}_{ts}.log"
	history_path = log_dir / f"{run_name}_{model_name}_{ts}_history.csv"
	plot_path = log_dir / f"{run_name}_{model_name}_{ts}_curves.png"
	verbose = bool(args.verbose or logging_cfg.get("verbose", False))
	logger = TrainLogger(log_path=log_path, verbose=verbose)

	logger.log(f"[run] config={args.config} device={device}", always=True)
	logger.log(f"[run] model={model_name}", always=True)
	logger.log(f"[run] window_size={window_size}", always=True)
	logger.log(f"[run] normalize_mode={resolve_normalize_mode(cfg)}", always=True)
	logger.log(f"[run] aoa_cache_root={aoa_root}", always=True)
	logger.log(f"[run] labels_root={labels_root}", always=True)
	logger.log(f"[run] val_env={args.val_env} test_env={args.test_env}", always=True)
	logger.log(f"[run] logs_file={log_path}", always=True)
	logger.log(
		"[run] "
		f"selection_mode={str(cfg.get('checkpoint', {}).get('selection', {}).get('mode', 'accuracy')).lower()} "
		f"lambda_var={float(cfg.get('loss', {}).get('lambda_var', 0.0)):.3f} "
		f"lambda_batch_div={float(cfg.get('loss', {}).get('lambda_batch_div', 0.0)):.3f} "
		f"lambda_inter_div={float(cfg.get('loss', {}).get('lambda_inter_div', 0.0)):.3f} "
		f"lambda_intra_div={float(cfg.get('loss', {}).get('lambda_intra_div', 0.0)):.3f}",
		always=True,
	)

	train_loader, val_loader, test_loader, split_stats = build_dataloaders(
		cfg=cfg,
		args=args,
		device=device,
		aoa_root=aoa_root,
		labels_root=labels_root,
		val_env=args.val_env,
		test_env=args.test_env,
		window_size=window_size,
	)
	logger.log(
		"[data] "
		f"dataset_size={split_stats['dataset_size']} train_size={split_stats['train_size']} "
		f"val_size={split_stats['val_size']} test_size={split_stats['test_size']} "
		f"batch_size={split_stats['batch_size']} window_size={split_stats['window_size']}",
		always=True,
	)

	model = build_model(cfg, device, args.model_name, window_size=window_size, num_envs=num_envs)
	action_aux_cfg = cfg.get("action_aux", {})
	use_action_aux = bool(action_aux_cfg.get("enable", False))
	lambda_action_cls = float(action_aux_cfg.get("lambda_cls", 0.0))
	action_head: ActionAuxHead | None = None
	if use_action_aux:
		feature_dim = int(getattr(model, "feature_dim"))
		action_head = ActionAuxHead(
			input_dim=feature_dim,
			num_actions=int(action_aux_cfg.get("num_actions", len(ACTION_NAMES))),
			dropout=float(action_aux_cfg.get("dropout", cfg.get("model", {}).get("dropout", 0.2))),
		).to(device)
		logger.log(
			f"[run] action_aux enabled=True lambda_cls={lambda_action_cls:.3f} num_actions={int(action_aux_cfg.get('num_actions', len(ACTION_NAMES)))}",
			always=True,
		)
	else:
		logger.log("[run] action_aux enabled=False", always=True)
	loss_cfg = cfg.get("loss", {})
	criterion = PoseStructureLoss(
		lambda_pose=float(loss_cfg.get("lambda_pose", 1.0)),
		lambda_rel=float(loss_cfg.get("lambda_angle", 1.0)),
		lambda_dist=float(loss_cfg.get("lambda_len", 0.5)),
		lambda_dir=float(loss_cfg.get("lambda_dir", 0.2)),
		lambda_var=float(loss_cfg.get("lambda_var", 0.1)),
		lambda_batch_div=float(loss_cfg.get("lambda_batch_div", 0.0)),
		lambda_inter_div=float(loss_cfg.get("lambda_inter_div", 0.0)),
		lambda_intra_div=float(loss_cfg.get("lambda_intra_div", 0.0)),
		preserve_ratio=float(loss_cfg.get("preserve_ratio", 0.6)),
		huber_beta=float(loss_cfg.get("huber_delta", 0.05)),
	)
	optimizer_name = train_cfg.get("optimizer", "AdamW")
	lr = float(train_cfg.get("lr", 3e-4))
	weight_decay = float(train_cfg.get("weight_decay", 1e-4))
	trainable_params = list(model.parameters())
	if action_head is not None:
		trainable_params.extend(action_head.parameters())
	if optimizer_name.lower() == "adamw":
		optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
	else:
		optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

	ckpt_path = Path(args.checkpoint)
	if not ckpt_path.is_absolute():
		ckpt_path = PROJECT_ROOT / ckpt_path
	last_ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_last{ckpt_path.suffix}")
	start_epoch = 0
	if args.resume and ckpt_path.exists():
		start_epoch = load_checkpoint(
			ckpt_path,
			model,
			optimizer,
			extra_modules={"action_head": action_head} if action_head is not None else None,
		)
		logger.log(f"[ckpt] resumed_from={ckpt_path} start_epoch={start_epoch+1}", always=True)

	epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 1))
	start_time = time.time()
	prev_elapsed = 0.0
	checkpoint_cfg = cfg.get("checkpoint", {})
	best_selection_score = float("-inf")
	best_val = float("inf")
	best_epoch = start_epoch
	history_rows: list[dict[str, Any]] = []

	for epoch in range(start_epoch, epochs):
		ep = epoch + 1
		current_batch_div_lambda = resolve_scheduled_lambda(loss_cfg, "lambda_batch_div", "batch_div_schedule", ep, epochs)
		current_inter_div_lambda = resolve_scheduled_lambda(loss_cfg, "lambda_inter_div", "inter_div_schedule", ep, epochs)
		current_intra_div_lambda = resolve_scheduled_lambda(loss_cfg, "lambda_intra_div", "intra_div_schedule", ep, epochs)
		criterion.lambda_batch_div = current_batch_div_lambda
		criterion.lambda_inter_div = current_inter_div_lambda
		criterion.lambda_intra_div = current_intra_div_lambda
		logger.log(
			f"[epoch_start] epoch={ep}/{epochs} train_size={split_stats['train_size']} val_size={split_stats['val_size']} "
			f"test_size={split_stats['test_size']} lr={lr:.3e} "
			f"lambda_batch_div={current_batch_div_lambda:.3f} lambda_inter_div={current_inter_div_lambda:.3f} "
			f"lambda_intra_div={current_intra_div_lambda:.3f}",
			always=True,
		)
		train_loss, train_stats = train_one_epoch(
			model=model,
			action_head=action_head,
			loader=train_loader,
			device=device,
			optimizer=optimizer,
			criterion=criterion,
			lambda_action_cls=lambda_action_cls,
			max_steps=args.max_steps,
			epoch=ep,
			num_epochs=epochs,
			logger=logger,
			log_interval=max(1, args.log_interval),
		)
		val_loss, val_nm, val_parts = evaluate(model, action_head, val_loader, device, criterion, lambda_action_cls)
		selection_score, selection_mode = compute_checkpoint_selection_score(val_nm, val_parts, checkpoint_cfg)

		elapsed = time.time() - start_time
		ep_time = elapsed - prev_elapsed
		prev_elapsed = elapsed
		eta_total = (elapsed / ep) * epochs if ep > 0 else 0.0
		eta = max(0.0, eta_total - elapsed)
		history_row = {
			"epoch": ep,
			"batch_div_lambda": float(current_batch_div_lambda),
			"inter_div_lambda": float(current_inter_div_lambda),
			"intra_div_lambda": float(current_intra_div_lambda),
			"selection_score": float(selection_score),
			"train_loss": float(train_loss),
			"val_loss": float(val_loss),
			"val_nmpjpe": float(val_nm),
			"train_pose_loss": float(train_stats["pose_loss_mean"]),
			"train_rel_loss": float(train_stats["rel_loss_mean"]),
			"train_dist_loss": float(train_stats["dist_loss_mean"]),
			"train_dir_loss": float(train_stats["dir_loss_mean"]),
			"train_var_loss": float(train_stats["var_loss_mean"]),
			"train_batch_div_loss": float(train_stats["batch_div_loss_mean"]),
			"train_inter_div_loss": float(train_stats["inter_div_loss_mean"]),
			"train_intra_div_loss": float(train_stats["intra_div_loss_mean"]),
			"val_pose_loss": float(val_parts["pose_loss"]),
			"val_rel_loss": float(val_parts["rel_loss"]),
			"val_dist_loss": float(val_parts["dist_loss"]),
			"val_dir_loss": float(val_parts["dir_loss"]),
			"val_var_loss": float(val_parts["var_loss"]),
			"val_batch_div_loss": float(val_parts["batch_div_loss"]),
			"val_inter_div_loss": float(val_parts["inter_div_loss"]),
			"val_intra_div_loss": float(val_parts["intra_div_loss"]),
			"train_pred_std_mean": float(train_stats["pred_std_mean"]),
			"train_target_std_mean": float(train_stats["target_std_mean"]),
			"train_std_ratio": float(train_stats["std_ratio_mean"]),
			"train_pred_pair_dist_mean": float(train_stats["pred_pair_dist_mean"]),
			"train_target_pair_dist_mean": float(train_stats["target_pair_dist_mean"]),
			"train_pair_dist_ratio": float(train_stats["pair_dist_ratio_mean"]),
			"train_pred_inter_pair_dist_mean": float(train_stats["pred_inter_pair_dist_mean"]),
			"train_target_inter_pair_dist_mean": float(train_stats["target_inter_pair_dist_mean"]),
			"train_inter_pair_dist_ratio": float(train_stats["inter_pair_dist_ratio_mean"]),
			"train_pred_intra_pair_dist_mean": float(train_stats["pred_intra_pair_dist_mean"]),
			"train_target_intra_pair_dist_mean": float(train_stats["target_intra_pair_dist_mean"]),
			"train_intra_pair_dist_ratio": float(train_stats["intra_pair_dist_ratio_mean"]),
			"val_pred_std_mean": float(val_parts["pred_std_mean"]),
			"val_target_std_mean": float(val_parts["target_std_mean"]),
			"val_std_ratio": float(val_parts["std_ratio"]),
			"val_pred_pair_dist_mean": float(val_parts["pred_pair_dist_mean"]),
			"val_target_pair_dist_mean": float(val_parts["target_pair_dist_mean"]),
			"val_pair_dist_ratio": float(val_parts["pair_dist_ratio"]),
			"val_pred_inter_pair_dist_mean": float(val_parts["pred_inter_pair_dist_mean"]),
			"val_target_inter_pair_dist_mean": float(val_parts["target_inter_pair_dist_mean"]),
			"val_inter_pair_dist_ratio": float(val_parts["inter_pair_dist_ratio"]),
			"val_pred_intra_pair_dist_mean": float(val_parts["pred_intra_pair_dist_mean"]),
			"val_target_intra_pair_dist_mean": float(val_parts["target_intra_pair_dist_mean"]),
			"val_intra_pair_dist_ratio": float(val_parts["intra_pair_dist_ratio"]),
			"train_action_cls_loss": float(train_stats["action_cls_loss_mean"]),
			"train_action_acc": float(train_stats["action_acc_mean"]),
			"val_action_cls_loss": float(val_parts["action_cls_loss"]),
			"val_action_acc": float(val_parts["action_acc"]),
			"grad_norm_mean": float(train_stats["grad_norm_mean"]),
			"nan_loss": bool(train_stats["has_nan_loss"]),
			"nan_grad": bool(train_stats["has_nan_grad"]),
		}
		history_rows.append(history_row)
		save_history_csv(history_path, history_rows)

		logger.log(
			"[epoch_summary] "
			f"epoch={ep}/{epochs} time_epoch={ep_time:.1f}s elapsed={elapsed:.1f}s eta={eta:.1f}s "
			f"lambda_batch_div={current_batch_div_lambda:.3f} lambda_inter_div={current_inter_div_lambda:.3f} lambda_intra_div={current_intra_div_lambda:.3f} "
			f"selection_mode={selection_mode} selection_score={selection_score:.6f} "
			f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_nmpjpe={val_nm:.6f} "
			f"train_pose={train_stats['pose_loss_mean']:.6f} train_rel={train_stats['rel_loss_mean']:.6f} train_dist={train_stats['dist_loss_mean']:.6f} train_dir={train_stats['dir_loss_mean']:.6f} train_var={train_stats['var_loss_mean']:.6f} train_batch_div={train_stats['batch_div_loss_mean']:.6f} train_inter_div={train_stats['inter_div_loss_mean']:.6f} train_intra_div={train_stats['intra_div_loss_mean']:.6f} "
			f"val_pose={val_parts['pose_loss']:.6f} val_rel={val_parts['rel_loss']:.6f} val_dist={val_parts['dist_loss']:.6f} val_dir={val_parts['dir_loss']:.6f} val_var={val_parts['var_loss']:.6f} val_batch_div={val_parts['batch_div_loss']:.6f} val_inter_div={val_parts['inter_div_loss']:.6f} val_intra_div={val_parts['intra_div_loss']:.6f} "
			f"train_std={train_stats['pred_std_mean']:.6f}/{train_stats['target_std_mean']:.6f} train_std_ratio={train_stats['std_ratio_mean']:.6f} "
			f"val_std={val_parts['pred_std_mean']:.6f}/{val_parts['target_std_mean']:.6f} val_std_ratio={val_parts['std_ratio']:.6f} "
			f"train_pair_dist={train_stats['pred_pair_dist_mean']:.6f}/{train_stats['target_pair_dist_mean']:.6f} train_pair_ratio={train_stats['pair_dist_ratio_mean']:.6f} "
			f"val_pair_dist={val_parts['pred_pair_dist_mean']:.6f}/{val_parts['target_pair_dist_mean']:.6f} val_pair_ratio={val_parts['pair_dist_ratio']:.6f} "
			f"train_inter_pair={train_stats['pred_inter_pair_dist_mean']:.6f}/{train_stats['target_inter_pair_dist_mean']:.6f} train_inter_ratio={train_stats['inter_pair_dist_ratio_mean']:.6f} "
			f"val_inter_pair={val_parts['pred_inter_pair_dist_mean']:.6f}/{val_parts['target_inter_pair_dist_mean']:.6f} val_inter_ratio={val_parts['inter_pair_dist_ratio']:.6f} "
			f"train_intra_pair={train_stats['pred_intra_pair_dist_mean']:.6f}/{train_stats['target_intra_pair_dist_mean']:.6f} train_intra_ratio={train_stats['intra_pair_dist_ratio_mean']:.6f} "
			f"val_intra_pair={val_parts['pred_intra_pair_dist_mean']:.6f}/{val_parts['target_intra_pair_dist_mean']:.6f} val_intra_ratio={val_parts['intra_pair_dist_ratio']:.6f} "
			f"train_action_cls={train_stats['action_cls_loss_mean']:.6f} train_action_acc={train_stats['action_acc_mean']:.3f} "
			f"val_action_cls={val_parts['action_cls_loss']:.6f} val_action_acc={val_parts['action_acc']:.3f} "
			f"grad_norm_mean={train_stats['grad_norm_mean']:.3f} "
			f"nan_loss={train_stats['has_nan_loss']} nan_grad={train_stats['has_nan_grad']} "
			f"actions_covered={train_stats['actions_covered']} envs_covered={train_stats['envs_covered']}",
			always=True,
		)
		save_checkpoint(
			last_ckpt_path,
			model,
			optimizer,
			epoch=ep,
			cfg=cfg,
			extra_modules={"action_head": action_head} if action_head is not None else None,
		)
		logger.log(f"[ckpt] saved_last path={last_ckpt_path} epoch={ep}", always=True)

		if selection_score > best_selection_score:
			best_selection_score = selection_score
			best_val = val_nm
			best_epoch = ep
			save_checkpoint(
				ckpt_path,
				model,
				optimizer,
				epoch=ep,
				cfg=cfg,
				extra_modules={"action_head": action_head} if action_head is not None else None,
			)
			logger.log(
				f"[ckpt] saved path={ckpt_path} epoch={ep} best_selection_score={best_selection_score:.6f} "
				f"val_nmpjpe={best_val:.6f}",
				always=True,
			)
		else:
			logger.log(
				f"[ckpt] no_improve best_epoch={best_epoch} best_selection_score={best_selection_score:.6f} "
				f"best_val_nmpjpe={best_val:.6f}",
				always=True,
			)

	load_checkpoint(
		ckpt_path,
		model,
		extra_modules={"action_head": action_head} if action_head is not None else None,
	)
	train_loss, train_nm, train_parts = evaluate(model, action_head, train_loader, device, criterion, lambda_action_cls)
	test_loss, test_nm, test_parts = evaluate(model, action_head, test_loader, device, criterion, lambda_action_cls)
	logger.log(f"[train eval] nMPJPE={train_nm:.4f}", always=True)
	logger.log(f"[test eval] nMPJPE={test_nm:.4f}", always=True)
	plot_history(plot_path, history_rows, logger)
	logger.log(f"[history] saved path={history_path}", always=True)
	assessment, reason = assess_fit(history_rows, test_nm)
	logger.log(
		f"[test_summary] best_epoch={best_epoch} test_loss={test_loss:.6f} test_nmpjpe={test_nm:.6f} "
		f"test_pose={test_parts['pose_loss']:.6f} test_rel={test_parts['rel_loss']:.6f} test_dist={test_parts['dist_loss']:.6f} "
		f"test_dir={test_parts['dir_loss']:.6f} test_var={test_parts['var_loss']:.6f} test_batch_div={test_parts['batch_div_loss']:.6f} "
		f"test_inter_div={test_parts['inter_div_loss']:.6f} test_intra_div={test_parts['intra_div_loss']:.6f} "
		f"test_std={test_parts['pred_std_mean']:.6f}/{test_parts['target_std_mean']:.6f} test_std_ratio={test_parts['std_ratio']:.6f} "
		f"test_pair_dist={test_parts['pred_pair_dist_mean']:.6f}/{test_parts['target_pair_dist_mean']:.6f} test_pair_ratio={test_parts['pair_dist_ratio']:.6f} "
		f"test_inter_pair={test_parts['pred_inter_pair_dist_mean']:.6f}/{test_parts['target_inter_pair_dist_mean']:.6f} test_inter_ratio={test_parts['inter_pair_dist_ratio']:.6f} "
		f"test_intra_pair={test_parts['pred_intra_pair_dist_mean']:.6f}/{test_parts['target_intra_pair_dist_mean']:.6f} test_intra_ratio={test_parts['intra_pair_dist_ratio']:.6f} "
		f"test_action_cls={test_parts['action_cls_loss']:.6f} test_action_acc={test_parts['action_acc']:.3f}",
		always=True,
	)
	logger.log(f"[assessment] status={assessment} reason={reason}", always=True)


if __name__ == "__main__":
	main()
