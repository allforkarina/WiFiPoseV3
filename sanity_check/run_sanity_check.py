"""
Architecture Trainability Sanity Check
=======================================
Verifies the full training pipeline (forward → loss → backward → optimizer update)
using PURE GAUSSIAN NOISE inputs and RANDOM BINARY labels.

Expected result:
  - val_acc ≈ 45–55%  (coin-flip level, task is unlearnable by design)
  - loss   ≈ ln(2) ≈ 0.693  (BCEWithLogitsLoss theoretical minimum on random labels)
  - grad_norm non-zero, nan_flag = False throughout

Usage (Windows & Linux):
  python sanity_check/run_sanity_check.py
  python sanity_check/run_sanity_check.py --epochs 5 --n_samples 2000 --device cpu
  python sanity_check/run_sanity_check.py --grad_clip --label_mode sign
"""

from __future__ import annotations

import argparse
import math
import sys
import os

# ---------------------------------------------------------------------------
# Make sure the project root (one level up from this file) is on sys.path so
# that "from mymodels import ConvBaseline" works regardless of cwd.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from mymodels import ConvBaseline

# ──────────────────────────────────────────────
#  1.  Pseudo Dataset
# ──────────────────────────────────────────────

class RandomNoiseBinaryDataset(Dataset):
    """Pure Gaussian-noise inputs with random binary labels.

    X and y are generated from INDEPENDENT random streams, so there is
    zero learnable structure by construction.

    Args:
        n_samples:   total number of samples
        input_shape: shape per sample, default (1, 181) matching real AOA input
        label_mode:  'bernoulli' → Bernoulli(0.5) labels  |
                     'sign'     → sign(N(0,1)) → {0,1}
        seed:        RNG seed for reproducibility
    """

    def __init__(
        self,
        n_samples: int = 2000,
        input_shape: tuple = (1, 181),
        label_mode: str = "bernoulli",
        seed: int = 42,
    ) -> None:
        super().__init__()

        rng_x = torch.Generator()
        rng_x.manual_seed(seed)

        rng_y = torch.Generator()
        rng_y.manual_seed(seed + 9999)  # deliberately different seed → independent

        # X ~ N(0, 1)  shape (n_samples, *input_shape)
        self.X = torch.randn(n_samples, *input_shape, generator=rng_x).float()

        # y: random 0/1 labels, independent of X
        if label_mode == "bernoulli":
            self.y = torch.bernoulli(
                torch.full((n_samples,), 0.5), generator=rng_y
            ).long()
        elif label_mode == "sign":
            self.y = (torch.randn(n_samples, generator=rng_y) > 0).long()
        else:
            raise ValueError(f"Unknown label_mode: '{label_mode}'. "
                             "Choose 'bernoulli' or 'sign'.")

        pos_ratio = self.y.float().mean().item()
        print(f"[Dataset] n={n_samples}  shape={input_shape}  "
              f"label_mode={label_mode}  pos_ratio={pos_ratio:.3f}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx].float()  # float for BCEWithLogitsLoss


# ──────────────────────────────────────────────
#  2.  Model Wrapper  (binary classification head)
# ──────────────────────────────────────────────

class SanityCheckModel(nn.Module):
    """Wraps the existing ConvBaseline backbone with a temporary 1-output
    classification head.  The backbone is NOT modified.

    Architecture (inference path):
        x (B,1,181)
          → backbone.feature  (3× Conv1d+BN+ReLU → AdaptiveAvgPool1d)
          → squeeze(-1)        (B, 256)
          → backbone.dropout
          → cls_head           (B, 1)     ← new, for this test only

    Loss: BCEWithLogitsLoss (logit output; numerically more stable than
          Sigmoid + BCELoss because it fuses them in log-sum-exp form).
    """

    def __init__(self, backbone: ConvBaseline) -> None:
        super().__init__()
        self.feature = backbone.feature
        self.dropout = backbone.dropout
        hidden_dim = backbone.head.in_features  # 256 by default
        self.cls_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:            # (B, 181) → (B, 1, 181)
            x = x.unsqueeze(1)
        feats = self.feature(x).squeeze(-1)   # (B, 256)
        feats = self.dropout(feats)
        return self.cls_head(feats).squeeze(1)  # (B,)  logits


# ──────────────────────────────────────────────
#  3.  Gradient Norm Helper
# ──────────────────────────────────────────────

def get_grad_norms(model: SanityCheckModel) -> dict:
    """Return gradient norms for the first conv layer and the cls_head.

    Also checks for NaN / Inf in any gradient in the model.
    Must be called AFTER loss.backward() and BEFORE optimizer.step().
    """

    def _norm(layer: nn.Module) -> float:
        total = 0.0
        for p in layer.parameters():
            if p.grad is not None:
                total += p.grad.detach().norm().item() ** 2
        return math.sqrt(total)

    def _has_bad_grad(model: nn.Module) -> bool:
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    return True
        return False

    first_conv = model.feature[0]   # nn.Conv1d(1, 64, ...)
    return {
        "first_conv": _norm(first_conv),
        "cls_head":   _norm(model.cls_head),
        "nan_flag":   _has_bad_grad(model),
    }


# ──────────────────────────────────────────────
#  4.  Train / Eval helpers
# ──────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip: float | None,
) -> tuple[float, float, dict | None]:
    """One train or eval epoch.

    Returns (avg_loss, accuracy_pct, grad_info_last_batch).
    grad_info is None when optimizer is None (eval mode).
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0
    last_grad_info = None

    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)               # (B,)
            loss = criterion(logits, y_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()

                last_grad_info = get_grad_norms(model)

                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

            # accuracy: threshold logits at 0  (equiv. to prob > 0.5)
            preds = (logits.detach() > 0).long()
            correct += (preds == y_batch.long()).sum().item()
            total += y_batch.size(0)
            total_loss += loss.item() * y_batch.size(0)

    avg_loss = total_loss / total if total else 0.0
    acc = correct / total * 100.0 if total else 0.0
    return avg_loss, acc, last_grad_info


# ──────────────────────────────────────────────
#  5.  Decision Table
# ──────────────────────────────────────────────

def print_decision_table(val_accs: list[float], nan_ever: bool) -> str:
    """Analyse results and print a structured diagnostic table."""

    final_acc = val_accs[-1] if val_accs else 0.0
    mean_acc  = sum(val_accs) / len(val_accs) if val_accs else 0.0
    unstable  = (max(val_accs) - min(val_accs)) > 15 if len(val_accs) >= 3 else False

    sep = "=" * 70
    print(f"\n{sep}")
    print("  SANITY CHECK — RESULT DECISION TABLE")
    print(sep)
    print(f"  final val_acc : {final_acc:.1f}%")
    print(f"  mean  val_acc : {mean_acc:.1f}%")
    print(f"  unstable      : {'Yes' if unstable else 'No'} "
          f"(range {max(val_accs,default=0):.1f}%–{min(val_accs,default=0):.1f}%)")
    print(f"  nan_ever      : {'Yes ⚠' if nan_ever else 'No'}")
    print(sep)

    verdict = "UNKNOWN"

    if 45.0 <= mean_acc <= 55.0 and not nan_ever:
        verdict = "PASS"
        print("  ✅  PASS  — val_acc ≈ 50%  (coin-flip).")
        print("       Pipeline is correct.  Model cannot learn from pure noise.")

    if mean_acc > 65.0:
        verdict = "FAIL:LEAK"
        print("  ⚠️  val_acc too HIGH (> 65%).")
        print("     Possible causes & actions:")
        print("     1. Label-input correlation: re-check label_mode & seed independence")
        print("     2. n_samples too small: increase --n_samples (≥ 2000 recommended)")
        print("     3. Overfitting noise: reduce --batch_size or add Dropout")
        print("     4. Check RNG: ensure X and y use distinct Generator seeds")

    if mean_acc < 40.0:
        verdict = "FAIL:OPTIM"
        print("  ⚠️  val_acc too LOW (< 40%).")
        print("     Possible causes & actions:")
        print("     1. Loss implementation: confirm BCEWithLogitsLoss receives logits (no Sigmoid beforehand)")
        print("     2. Learning rate: try adjusting --lr (default 3e-4)")
        print("     3. Optimizer: verify AdamW is applied to all trainable params")
        print("     4. Check if loss is actually decreasing across epochs (not stuck)")

    if unstable and final_acc < 45.0:
        verdict = "FAIL:EXPLODE"
        print("  ⚠️  Accuracy unstable and dropping — suspect gradient explosion.")
        print("     Actions:")
        print("     1. Enable gradient clipping:  --grad_clip")
        print("     2. Lower learning rate:        --lr 1e-4 or smaller")
        print("     3. Check BatchNorm after each Conv (should stabilise activations)")

    if nan_ever:
        verdict = "FAIL:NAN"
        print("  ⚠️  NaN/Inf detected in gradients.")
        print("     Actions:")
        print("     1. Enable gradient clipping:  --grad_clip")
        print("     2. Check for division-by-zero or log(0) in custom layers")
        print("     3. Lower learning rate significantly")

    print(sep)

    # Grad norm ≈ 0 advisory (no separate verdict, just a hint)
    print()
    return verdict


# ──────────────────────────────────────────────
#  6.  Main
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Architecture trainability sanity check — random noise binary classification."
    )
    p.add_argument("--epochs",     type=int,   default=5,      help="Number of training epochs (default: 5)")
    p.add_argument("--batch_size", type=int,   default=64,     help="Batch size (default: 64, matches real training)")
    p.add_argument("--lr",         type=float, default=3e-4,   help="Learning rate (default: 3e-4)")
    p.add_argument("--n_samples",  type=int,   default=2000,   help="Total pseudo-dataset size (default: 2000)")
    p.add_argument("--val_ratio",  type=float, default=0.2,    help="Fraction of data used for validation (default: 0.2)")
    p.add_argument("--label_mode", type=str,   default="bernoulli", choices=["bernoulli", "sign"],
                   help="Label generation mode: 'bernoulli' (default) or 'sign'")
    p.add_argument("--grad_clip",  action="store_true",        help="Enable gradient clipping at norm=1.0 (default: off)")
    p.add_argument("--clip_value", type=float, default=1.0,    help="Gradient clip max norm (default: 1.0)")
    p.add_argument("--seed",       type=int,   default=42,     help="RNG seed (default: 42)")
    p.add_argument("--device",     type=str,   default="auto", help="Device: 'cpu', 'cuda', or 'auto' (default: auto)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── device ──────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ── reproducibility ─────────────────────────────
    torch.manual_seed(args.seed)

    # ── print header ────────────────────────────────
    val_size  = int(args.n_samples * args.val_ratio)
    train_size = args.n_samples - val_size
    print("=" * 70)
    print("  ARCHITECTURE TRAINABILITY SANITY CHECK")
    print("=" * 70)
    print(f"  device     = {device}")
    print(f"  n_samples  = {args.n_samples}  (train={train_size}, val={val_size})")
    print(f"  batch_size = {args.batch_size}")
    print(f"  lr         = {args.lr}")
    print(f"  epochs     = {args.epochs}")
    print(f"  label_mode = {args.label_mode}")
    print(f"  grad_clip  = {'ON (max_norm=' + str(args.clip_value) + ')' if args.grad_clip else 'OFF'}")
    print(f"  seed       = {args.seed}")
    print(f"  ln(2)      = {math.log(2):.4f}  ← expected loss for random binary labels")
    print("=" * 70)

    # ── dataset & loaders ───────────────────────────
    dataset = RandomNoiseBinaryDataset(
        n_samples=args.n_samples,
        input_shape=(1, 181),      # matches real AOA input (C=1, L=181)
        label_mode=args.label_mode,
        seed=args.seed,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # num_workers=0 for Windows compatibility (avoids multiprocessing pickling issues)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── model ───────────────────────────────────────
    backbone = ConvBaseline(
        input_channels=1,
        input_length=181,
        hidden_dim=256,
        num_joints=17,
        out_dim=2,
        dropout=0.2,
    )
    model = SanityCheckModel(backbone).to(device)

    cls_params = list(model.cls_head.parameters())  # only cls_head is new
    backbone_params = [p for p in model.parameters() if not any(p is q for q in cls_params)]

    # All params are trainable — this tests the full gradient flow
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    grad_clip_value = args.clip_value if args.grad_clip else None

    # ── training loop ───────────────────────────────
    val_accs: list[float] = []
    nan_ever = False

    col = "{:>5} | {:>8} | {:>10} | {:>9} | {:>20} | {:>17} | {:>8}"
    header = col.format("Epoch", "Loss", "Train Acc", "Val Acc",
                        "GradNorm(first_conv)", "GradNorm(cls)", "NaN")
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, grad_info = run_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip_value
        )
        _, val_acc, _ = run_epoch(
            model, val_loader, criterion, None, device, None
        )

        val_accs.append(val_acc)

        gn_first = grad_info["first_conv"] if grad_info else float("nan")
        gn_cls   = grad_info["cls_head"]   if grad_info else float("nan")
        nan_flag = grad_info["nan_flag"]   if grad_info else False
        if nan_flag:
            nan_ever = True

        nan_str = "⚠️ YES" if nan_flag else "No"
        row = col.format(
            f"{epoch:02d}/{args.epochs:02d}",
            f"{train_loss:.4f}",
            f"{train_acc:.1f}%",
            f"{val_acc:.1f}%",
            f"{gn_first:.6f}",
            f"{gn_cls:.6f}",
            nan_str,
        )
        print(row)

    # ── decision table ──────────────────────────────
    verdict = print_decision_table(val_accs, nan_ever)

    # ── grad-norm zero advisory ─────────────────────
    print("  Grad-norm = 0 advisory:")
    print("    If GradNorm(first_conv) is consistently < 1e-7, gradient is not")
    print("    flowing to the early layers.  Possible fixes:")
    print("    1. Verify BatchNorm is not zeroing activations (check running stats)")
    print("    2. Simplify the network (remove layers) and re-run")
    print("    3. Try a smaller learning rate + more epochs")
    print()

    # exit code: 0 = PASS, 1 = any FAIL category
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
