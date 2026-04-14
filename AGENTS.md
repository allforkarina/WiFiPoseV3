# Repository Guidelines

## Project Structure & Module Organization
Core entry points are `train.py`, `eval.py`, and `diagnose_pose_collapse.py`. Model code lives in `mymodels/`, datasets in `dataloader/`, preprocessing in `preprocess/`, and shared utilities in `utils/`. Runtime configs live under `configs/`. Formal experiment logs are stored in `logs/`; temporary Windows smoke artifacts must stay under `tmp/`; checkpoints stay under `checkpoints/`.

## Build, Test, and Development Commands
Run commands from the repository root.

- `python train.py --config configs/default.yaml` runs the default Windows-facing training config.
- `python tools/run_windows_smoke.py --track non_dann` runs the fixed local non-DANN smoke path.
- `python tools/run_windows_smoke.py --track non_dann --variant short` runs the short-run early-overfit smoke path.
- `python tools/run_windows_smoke.py --track non_dann --variant short_reg` runs the stronger-regularization smoke path.
- `python tools/run_windows_smoke.py --track non_dann --variant short_reg_aug` runs the stronger-regularization plus augmentation smoke path.
- `python tools/run_windows_smoke.py --track dann` runs the fixed local DANN smoke path.
- `python tools/run_linux_formal.py --track non_dann --variant accuracy` runs the formal non-DANN accuracy-first track.
- `python tools/run_linux_formal.py --track non_dann --variant short` runs the short-run formal non-DANN track.
- `python tools/run_linux_formal.py --track non_dann --variant short_reg` runs the stronger-regularization formal non-DANN track.
- `python tools/run_linux_formal.py --track non_dann --variant short_reg_aug` runs the stronger-regularization plus augmentation formal non-DANN track.
- `python tools/run_linux_formal.py --track non_dann --variant balanced` runs the formal non-DANN balanced-selection track.
- `python tools/run_linux_formal.py --track dann --variant accuracy` runs the formal DANN accuracy-first track.
- `python tools/run_linux_formal.py --track dann --variant balanced` runs the formal DANN balanced-selection track.
- `python eval.py --config configs/linux_non_dann.yaml --checkpoint checkpoints/<run>.pth` evaluates a saved checkpoint.
- `python sanity_check/run_sanity_check.py --epochs 5 --device cpu` verifies the minimal forward/backward pipeline.
- `python tools/prune_run_artifacts.py --keep 5` trims old tracked logs and checkpoints.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, focused functions, explicit argument names, and `Path`-based filesystem handling where practical. Use `snake_case` for variables, functions, config keys, and CLI flags. Use `PascalCase` for classes. Preserve surrounding style when touching mixed-format legacy files.

## Testing Guidelines
This repository uses script-level validation instead of a formal `pytest` suite. On Windows, only run lightweight smoke validation: config load, dataset creation, one or two batches of forward/loss/backward, and checkpoint/log path checks. Do not run full experiments locally. Full training, evaluation, and official metrics must run on the Linux server after `git pull`.

## Commit & Pull Request Guidelines
Use short imperative commit messages scoped to one change, such as `Add smoke config and formal track split`. Every meaningful change must update this `AGENTS.md`, then be synced with `git add`, `git commit`, and `git push`. Include tracked `logs/` when the Linux server produces a formal result worth preserving.

## Configuration & Data Notes
`configs/default.yaml` and `configs/linux.yaml` remain backward-compatible entry configs. Preferred current configs are:

- `configs/windows_smoke.yaml` for local smoke-only validation
- `configs/windows_smoke_short.yaml`, `configs/windows_smoke_short_reg.yaml`, and `configs/windows_smoke_short_reg_aug.yaml` for the early-overfit smoke matrix
- `configs/windows_smoke_dann.yaml` for local DANN smoke validation
- `configs/linux_non_dann_accuracy.yaml`, `configs/linux_non_dann_balanced.yaml`, and `configs/linux_non_dann.yaml` for the non-DANN matrix
- `configs/linux_non_dann_short_accuracy.yaml`, `configs/linux_non_dann_short_reg_accuracy.yaml`, and `configs/linux_non_dann_short_reg_aug_accuracy.yaml` for the non-DANN early-overfit mitigation matrix
- `configs/linux_dann_accuracy.yaml`, `configs/linux_dann_balanced.yaml`, and `configs/linux_dann.yaml` for the DANN matrix

Keep `domain_adaptation.use_dann` disabled unless the active experiment is explicitly the DANN track.

## Agent-Specific Instructions
- All user-facing replies, progress updates, and summaries must be written in Chinese.
- All content inside `AGENTS.md` must be written in English.
- Before any test or validation command, verify the runtime environment is `WiFiPose`; activate it first if needed.
- Windows is for code changes and smoke validation only. Linux is the only source of official full-training conclusions.
- After every meaningful code, config, or workflow change, update this file so the current optimization state remains handoff-ready.

## Current Optimization Targets
- **Completed**: Temporal Difference is integrated in the AoA dataset pipeline.
- **Completed**: `ResNet1DPose` already uses the Deep MLP head (`Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm -> GELU -> Linear(34)`).
- **Completed**: The formal pre-DANN baseline has been run on Linux (`train20260412_2206.log`) with `mean_rms + action_aux + Deep MLP head + diversity_first`.
- **Completed**: The project is no longer primarily blocked by average-pose collapse. The active bottleneck is poor cross-environment generalization.
- **Completed**: The workflow is now split into a Windows smoke path (`configs/windows_smoke.yaml`, `tools/run_windows_smoke.py`) and explicit Linux formal tracks (`configs/linux_non_dann.yaml`, `configs/linux_dann.yaml`, `tools/run_linux_formal.py`).
- **Completed**: Track A code support is in place. The non-DANN line now has explicit `accuracy`, `balanced`, and `diversity` checkpoint-selection variants.
- **Completed**: Track B code support is in place. The DANN line now mirrors the same `accuracy`, `balanced`, and `diversity` checkpoint-selection variants, and Windows can smoke-test the DANN path locally.
- **Completed**: `eval.py` has been re-aligned with the current `train.py` interfaces so the formal evaluation path is runnable again.
- **Completed**: The six-run Linux comparison matrix has been pulled back into `logs/train` and reviewed on Windows.
- **Completed**: `accuracy` is the only checkpoint-selection rule that remains competitive on final test `nMPJPE`. `balanced` and `diversity` improve `std_ratio` but consistently worsen pose accuracy.
- **Completed**: The matched `accuracy` comparison shows no meaningful DANN gain under the current setup. `linux_non_dann_accuracy_resnet1d_20260412-193627.log` remains the official best run with final test `nMPJPE=0.1900`, while `linux_dann_accuracy_resnet1d_20260413-031611.log` finishes at `0.1902`.
- **Completed**: Across the reviewed matrices, the best validation `nMPJPE` appears at epoch 1, which indicates the current training recipe overfits cross-environment generalization almost immediately.
- **Pending / Next Stage**: Lock `non-DANN + accuracy` as the official baseline and use it as the default comparison target for future ablations.
- **Completed**: The training loop now supports opt-in epoch-level LR scheduling, opt-in early stopping, runtime train-only AoA augmentation, and explicit overfit diagnostics (`lr`, `best_val_epoch`, `epochs_since_best`, `val_gap_from_best`, final `[overfit]` summary).
- **Pending / Next Stage**: Run the first four-run early-overfit mitigation matrix on Linux: `baseline`, `short`, `short_reg`, `short_reg_aug`.
- **Pending / Next Stage**: Treat `short` as the schedule-only ablation, `short_reg` as the regularization ablation, and `short_reg_aug` as the strongest low-risk training-side ablation against the `0.1900` official baseline.
- **Pending / Diagnostic**: Verify why `linux_dann_diversity_resnet1d_20260413-083219.log` currently ends at epoch 64 without final `train/test eval` lines before treating that run as a formal completed result.

## Testing and Validation Plan
- Windows validation must use the `windows_smoke*.yaml` configs and should only touch temporary outputs under `tmp/windows_smoke/`.
- Every Linux formal run must produce tracked `logs/` and be compared with the current official baseline `linux_non_dann_accuracy_resnet1d_20260412-193627.log` (`test nMPJPE=0.1900`).
- The default formal checkpoint-selection rule for official comparisons is now `accuracy`.
- `balanced` and `diversity` may still be run as diagnostics, but they must not be treated as primary selection rules unless they also improve final `nMPJPE`.
- The primary acceptance metric is always `val/test nMPJPE`.
- `std_ratio`, `action_aux` validation accuracy, and domain-classifier metrics are secondary diagnostics and must not override worse pose accuracy.
- For the current early-overfit stage, every formal log should also report `best_val_epoch`, `first_degradation_epoch`, `final_val_gap`, and whether early stopping triggered.
- Any future DANN formal run must be considered incomplete if the log does not include final `train eval`, `test eval`, and assessment lines.

## Current Workflow
1. Update code or configs on Windows.
2. Run `python tools/run_windows_smoke.py --track <non_dann|dann> [--variant <baseline|short|short_reg|short_reg_aug>]` inside `WiFiPose`.
3. Update `AGENTS.md`, commit, and push.
4. Pull on Linux and run the chosen formal track, using `baseline`, `short`, `short_reg`, and `short_reg_aug` for the current non-DANN overfit-mitigation matrix and reserving `balanced` or `diversity` for diagnostics only.
5. Push formal `logs/` back to GitHub.
6. Pull the new logs on Windows, compare them against the `0.1900` official baseline, and update the next optimization target list.
