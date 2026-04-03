from __future__ import annotations

import argparse
import csv
import math
import time
import sys
import subprocess
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


def build_subset_loader(
        ds: AOASampleDataset,
        indices: list[int],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        shuffle: bool,
) -> DataLoader:
	subset = Subset(ds, indices)
	return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


def build_dataloaders(
	cfg: Dict[str, Any],
	device: torch.device,
	aoa_root: Path,
	labels_root: Path,
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

	dataset_cfg = cfg.get("dataset", {})
	batch_size = int(dataset_cfg.get("batch_size", 32))
	num_workers = int(dataset_cfg.get("num_workers", 0))
	pin_memory = bool(dataset_cfg.get("pin_memory", False)) and device.type == "cuda"
	
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
	train_loader = build_subset_loader(ds, train_indices, batch_size, num_workers, pin_memory, True)
	val_loader = build_subset_loader(ds, val_indices, batch_size, num_workers, pin_memory, False)
	test_loader = build_subset_loader(ds, test_indices, batch_size, num_workers, pin_memory, False)
	
	stats = {
		"dataset_size": len(ds),
		"train_size": len(train_indices),
		"val_size": len(val_indices),
		"test_size": len(test_indices),
		"batch_size": batch_size,
		"window_size": window_size,
		"train_envs": train_envs_seen,
	}
	return train_loader, val_loader, test_loader, stats


def build_model(cfg: Dict[str, Any], device: torch.device, window_size: int | None = None, num_envs: int = 0) -> nn.Module:
	mcfg = cfg.get("model", {})
	model_name = mcfg.get("name", "resnet1d").lower()
	effective_window = int(window_size if window_size is not None else cfg.get("dataset", {}).get("window_size", 1))
	common_kwargs = {
		"input_channels": effective_window if model_name in {"conv1d_baseline", "resnet1d", "ms_tcn_pose"} else mcfg.get("input_channels", 1),
		"input_length": mcfg.get("input_length", 181),
		"hidden_dim": mcfg.get("hidden_dim", 256),
		"num_joints": mcfg.get("output_joints", 17),
		"out_dim": mcfg.get("output_dim", 2),
		"dropout": mcfg.get("dropout", 0.2),
		"num_envs": num_envs,
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
	import math

	for step, (x, y, meta) in enumerate(loader):
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
				p = ((epoch - 1) * total_steps + step) / max(1, num_epochs * total_steps)
				alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
				domain_logits = model.forward_env(features, alpha)
				domain_loss = F.cross_entropy(domain_logits, env_targets)
				domain_acc = (domain_logits.argmax(dim=1) == env_targets).float().mean()
				loss = loss + (lambda_domain * domain_loss)
			except Exception as e:
				pass

		loss.backward()
		grad_norm = _compute_grad_norm(model, action_head)
		
		log_msg = f"[train] epoch={epoch}/{num_epochs} step={step+1}/{total_steps} loss={loss.item():.6f} grad_norm={grad_norm:.3f}"
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

	avg_loss = running_loss / max(1, total_steps)
	stats = {
		"pose_loss": running_pose_loss / max(1, total_steps),
		"std_ratio": running_std_ratio / max(1, total_steps),
		"action_cls_loss": running_action_cls_loss / max(1, total_steps),
		"action_acc": running_action_acc / max(1, total_steps),
		"domain_loss": running_domain_loss / max(1, total_steps),
		"domain_acc": running_domain_acc / max(1, total_steps),
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
		for x, y, meta in loader:
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

	train_loader, val_loader, test_loader, split_stats = build_dataloaders(
		cfg=cfg, device=device, aoa_root=aoa_root, labels_root=labels_root, window_size=window_size
	)
	logger.log(f"[data] train={split_stats['train_size']} val={split_stats['val_size']} test={split_stats['test_size']}", always=True)

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
	history_rows: list[dict[str, Any]] = []

	for epoch in range(start_epoch, epochs):
		ep = epoch + 1
		
		train_loss, train_stats = train_one_epoch(
			model, action_head, train_loader, device, optimizer, criterion,
			lambda_action_cls, ep, epochs, logger, log_interval=10,
			use_dann=use_dann, lambda_domain=lambda_domain, env_to_id=env_to_id
		)
		val_loss, val_nm, val_parts = evaluate(model, action_head, val_loader, device, criterion, lambda_action_cls, use_dann, env_to_id)
		selection_score, selection_mode = compute_checkpoint_selection_score(val_nm, val_parts['std_ratio'], checkpoint_cfg)

		history_rows.append({
			"epoch": ep,
			"train_loss": train_loss,
			"train_pose": train_stats["pose_loss"],
			"train_std_ratio": train_stats["std_ratio"],
			"train_acc": train_stats["action_acc"],
			"val_loss": val_loss,
			"val_nmpjpe": val_nm,
			"val_pose": val_parts["pose_loss"],
			"val_std_ratio": val_parts["std_ratio"],
			"val_acc": val_parts["action_acc"],
			"selection_score": selection_score
		})
		save_history_csv(history_path, history_rows)

		logger.log(
			f"[summary] Epoch {ep}/{epochs} | "
			f"Train Loss: {train_loss:.4f} (Pose: {train_stats['pose_loss']:.4f}, StdRatio: {train_stats['std_ratio']:.3f}, Acc: {train_stats['action_acc']:.2%}, D_Acc: {train_stats.get('domain_acc', 0.0):.2%}) | "
			f"Val   Loss: {val_loss:.4f} (nMPJPE: {val_nm:.4f}, StdRatio: {val_parts['std_ratio']:.3f}, Acc: {val_parts['action_acc']:.2%}, D_Acc: {val_parts.get('domain_acc', 0.0):.2%}) | "
			f"Score: {selection_score:.4f}",
			always=True
		)

		if ep > 0:
			save_checkpoint(last_ckpt_path, model, optimizer, ep, cfg, {"action_head": action_head})
		
		if selection_score > best_selection_score:
			best_selection_score = selection_score
			best_val = val_nm
			best_epoch = ep
			save_checkpoint(ckpt_path, model, optimizer, ep, cfg, {"action_head": action_head})
			logger.log(f"[ckpt] New best model saved (Score: {best_selection_score:.4f}, nMPJPE: {best_val:.4f})", always=True)

	load_checkpoint(ckpt_path, model, extra_modules={"action_head": action_head})
	train_loss, train_nm, train_parts = evaluate(model, action_head, train_loader, device, criterion, lambda_action_cls, use_dann, env_to_id)
	test_loss, test_nm, test_parts = evaluate(model, action_head, test_loader, device, criterion, lambda_action_cls, use_dann, env_to_id)
	
	logger.log(f"[train eval] nMPJPE={train_nm:.4f}", always=True)
	logger.log(f"[test eval] nMPJPE={test_nm:.4f} (Acc: {test_parts['action_acc']:.2%})", always=True)
	
	assessment, reason = assess_fit(history_rows, test_nm)
	logger.log(f"[assessment] {assessment}: {reason}", always=True)

	try:
		prune_cmd = [sys.executable, str(PROJECT_ROOT / "tools" / "prune_run_artifacts.py"), "--keep", "10"]
		subprocess.run(prune_cmd, check=True)
		logger.log("[prune] Kept the latest 10 artifacts.", always=True)
	except Exception as e:
		logger.log(f"[prune] Warn: Failed to prune {e}", always=True)

if __name__ == "__main__":
	main()
