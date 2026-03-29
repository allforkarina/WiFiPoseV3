# Repository Guidelines

## Project Structure & Module Organization
Core entry points are `train.py`, `eval.py`, and `diagnose_pose_collapse.py`. Models live in `mymodels/`, data loading in `dataloader/`, preprocessing in `preprocess/`, and helpers in `utils/`. Runtime configuration is centered in `configs/default.yaml`; notes belong in `docs/`. Generated artifacts should stay in `logs/`, `checkpoints/`, and local `data/` paths.

## Build, Test, and Development Commands
Use direct Python entry points from the repository root.

- `python train.py --config configs/default.yaml --model_name resnet1d --val_env env3 --test_env env4` runs training with the default config.
- `python eval.py --config configs/default.yaml --checkpoint checkpoints/<run>.pth` evaluates a saved checkpoint and writes plots under `logs/eval/`.
- `python diagnose_pose_collapse.py --config configs/default.yaml --checkpoint_glob "checkpoints/*.pth"` summarizes collapse-related metrics across checkpoints.
- `python tools/diagnose_input_pose_separability.py --aoa_cache_root <aoa_cache> --labels_root <labels_root>` compares AoA-distance and pose-distance consistency across normalization settings.
- `python sanity_check/run_sanity_check.py --epochs 5 --device cpu` performs a smoke test of forward, loss, backward, and optimizer update.
- `python tools/run_collapse_ablation.py --dry_run` previews ablation commands.
- `python tools/prune_run_artifacts.py --keep 5` removes old logs and checkpoints.

## Coding Style & Naming Conventions
Follow existing Python style: type hints where practical, small focused functions, and `Path`-based filesystem handling. Use `snake_case` for functions, variables, and CLI flags; use `PascalCase` for classes. Prefer 4-space indentation in new files, but preserve nearby style in touched files. No formatter or linter config is checked in.

## Testing Guidelines
This repository relies on script-level validation instead of a formal `pytest` suite. Before opening a PR, training changes should pass `sanity_check/run_sanity_check.py`, and model or data changes should be checked with `eval.py` or `diagnose_pose_collapse.py`. If you add automated tests, place them in `tests/` and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects such as `Add optimization principle diagram` and `Tune action aux anti-collapse weights`. Keep messages concise, action-first, and scoped to one change. PRs should describe behavior changes, list config or data path updates, and include metrics, logs, or plots when training behavior changes.

## Configuration & Data Notes
Treat `configs/default.yaml` as the source of truth for data roots, split settings, and training defaults. Do not hard-code machine-specific paths; use CLI flags such as `--aoa_cache_root` and `--labels_root`.

## Agent-Specific Instructions
- All user-facing dialogue, progress updates, and summaries must be written in Chinese.
- After every code or document update, sync the current branch to GitHub with a scoped commit and `git push`. If push fails, report the blocker in the reply.
- The required runtime environment for this project is the `WiFiPose` conda environment. Before running tests or validation commands, verify the active environment; if it is not `WiFiPose`, run `conda activate WiFiPose` first.
- After every meaningful change, update the `Current Optimization Targets` section in this file so the goal list stays current.

## Current Optimization Targets
- Completed: use the new `AOA_data` features as the active training input with fixed per-frame percentile normalization; the `resnet1d + mean_rms + selection_mode=accuracy` baseline recovered to `val_nMPJPE=0.1948` and `test_nMPJPE=0.1943` under the 8x100 baseline budget.
- Completed: validate a pure-accuracy recovery baseline using `mean_rms`, `selection_mode=accuracy`, and zero diversity/action-aux losses on the fixed `AOA_data` preprocessing pipeline.
- In progress: explain average-pose collapse with measurable evidence; `tools/diagnose_input_pose_separability.py` remains the primary check for AoA/pose distance consistency under the new fixed preprocessing.
- Pending: restore training and evaluation semantic consistency, especially whether `pelvis_torso` still carries regression risk relative to the now-recovered `mean_rms` baseline.
- Pending: strengthen repeatable validation so changes affecting data, loss, or checkpoints are checked with `sanity_check/run_sanity_check.py`, `eval.py`, `diagnose_pose_collapse.py`, or `tools/diagnose_input_pose_separability.py`.
- Pending: keep all validation and test execution aligned to the `WiFiPose` conda environment to avoid environment-dependent regressions.

## 下阶段任务
- 第一阶段：基于已恢复的纯精度基线，重新运行 `tools/diagnose_input_pose_separability.py`、`eval.py` 与 `diagnose_pose_collapse.py`，补齐固定 `AOA_data` 预处理下的诊断证据。
- 第二阶段：逐项重新加入 `lambda_inter_div`、`diversity_first` 与 `action_aux`，分别评估它们对坍缩问题的独立影响。
- 第三阶段：在固定 `AOA_data` 预处理下，再评估 `pelvis_torso` 是否仍然带来语义或性能回归。
- 第四阶段：将后续训练与评估统一保持在这套固定 `AOA_data` 预处理和 `WiFiPose` 环境下，并持续记录可复现结果。
