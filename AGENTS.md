# Repository Guidelines

## Project Structure & Module Organization
Core entry points are `train.py`, `eval.py`, and `diagnose_pose_collapse.py`. Model code lives in `mymodels/`, datasets in `dataloader/`, preprocessing in `preprocess/`, and shared utilities in `utils/`. Runtime configs live under `configs/`. Formal experiment logs are stored in `logs/`; temporary Windows smoke artifacts must stay under `tmp/`; checkpoints stay under `checkpoints/`.

## Build, Test, and Development Commands
Run commands from the repository root.

- `python train.py --config configs/default.yaml` runs the default Windows-facing training config.
- `python tools/run_windows_smoke.py` runs the fixed local smoke path with `configs/windows_smoke.yaml`.
- `python tools/run_linux_formal.py --track non_dann` runs the formal non-DANN Linux track.
- `python tools/run_linux_formal.py --track dann` runs the formal DANN Linux track.
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
- `configs/linux_non_dann.yaml` for the formal non-DANN track
- `configs/linux_dann.yaml` for the formal DANN track

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
- **Pending / Track A**: Improve the non-DANN line first by treating checkpoint selection as the main variable and optimizing for `val/test nMPJPE`, while keeping `std_ratio` and `action_aux` as diagnostics.
- **Pending / Track B**: Run a strict DANN-vs-non-DANN comparison using matched seeds, splits, budgets, and checkpoint rules.

## Testing and Validation Plan
- Windows validation must use `configs/windows_smoke.yaml` and should only touch temporary outputs under `tmp/windows_smoke/`.
- Every Linux formal run must produce tracked `logs/` and be compared with the current formal baseline.
- The primary acceptance metric is always `val/test nMPJPE`.
- `std_ratio`, `action_aux` validation accuracy, and domain-classifier metrics are secondary diagnostics and must not override worse pose accuracy.

## Current Workflow
1. Update code or configs on Windows.
2. Run `python tools/run_windows_smoke.py` inside `WiFiPose`.
3. Update `AGENTS.md`, commit, and push.
4. Pull on Linux and run the chosen formal track.
5. Push formal `logs/` back to GitHub.
6. Pull the new logs on Windows and update the next optimization target list.
