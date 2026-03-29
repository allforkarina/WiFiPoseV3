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
- Completed: clear stale `logs/` and `checkpoints/` outputs produced before the new `AOA_data` experiment cycle so subsequent validation starts from a clean artifact state.
- Completed: use the new `AOA_data` features as the active training input with fixed per-frame percentile normalization; the `resnet1d + mean_rms + selection_mode=accuracy` baseline recovered to `val_nMPJPE=0.1948` and `test_nMPJPE=0.1943` under the 8x100 baseline budget.
- Completed: establish a clean control baseline with `mean_rms`, `selection_mode=accuracy`, and zero diversity/action-aux losses on the fixed `AOA_data` preprocessing pipeline.
- Completed: rebuild the clean control cycle from scratch after log/checkpoint cleanup, including sanity check, 10-step smoke, full 8x100 baseline, `eval.py`, `diagnose_pose_collapse.py`, and `tools/diagnose_input_pose_separability.py`.
- In progress: restart the project workflow from a clean cycle organized into three stages: baseline testing, diagnostic validation, and anti-collapse optimization.
- In progress: explain residual average-pose collapse with measurable evidence; in the rebuilt control cycle the full baseline reached `val_nMPJPE=0.194662`, `test_nMPJPE=0.194259`, `eval mean_nMPJPE=0.197071`, but `diagnose_pose_collapse.py` still shows `variance_ratio_pred_over_target≈0.0708` on `env3` and `≈0.0696` on `env4`, with `mse_pred_to_target` still slightly worse than the global mean-pose baseline.
- Pending: strengthen repeatable validation so every change affecting data, loss, selection strategy, or checkpoints is checked with `sanity_check/run_sanity_check.py`, `eval.py`, `diagnose_pose_collapse.py`, and `tools/diagnose_input_pose_separability.py`.
- Pending: restore training and evaluation semantic consistency, especially whether `pelvis_torso` still carries regression risk relative to the recovered `mean_rms` baseline after anti-collapse terms are reintroduced.
- Pending: keep all validation and test execution aligned to the `WiFiPose` conda environment to avoid environment-dependent regressions.

## 测试 Plan
- 第一阶段：先重新建立干净的控制组实验，只使用 `resnet1d + mean_rms + selection_mode=accuracy + zero diversity/action_aux`，重新产出 smoke、短程训练与完整 baseline 的最新结果。
- 第二阶段：每个候选改动只允许变动一个因素，实验顺序固定为 `lambda_inter_div` -> `selection_mode=diversity_first` -> `action_aux`。
- 第三阶段：每次实验统一保存训练日志、history、checkpoint 与评估输出，禁止未完成验证就继续叠加下一个改动。

## 验证 Plan
- 第一阶段：所有实验至少执行 `sanity_check/run_sanity_check.py` 或等价 smoke，确认前向、反向、优化器更新和数据加载正常。
- 第二阶段：每轮实验训练后统一运行 `eval.py`、`diagnose_pose_collapse.py` 与 `tools/diagnose_input_pose_separability.py`，分别检查精度、坍缩程度与输入可分性。
- 第三阶段：统一对照控制组指标 `val_nMPJPE≈0.1948`、`test_nMPJPE≈0.1943`、`variance_ratio≈0.057`、`mse_pred_to_target` 与 `mse_meanpose_to_target` 的关系。
- 第四阶段：只有当改动在验证集和测试集上都成立，且结论可复现，才允许进入下一轮优化。

## 优化 Plan
- 第一阶段：优先提升跨样本和跨动作多样性，核心目标是在不明显破坏 `nMPJPE` 的前提下抬升 `variance_ratio_pred_over_target`、`pred_group_std_mean` 与动作间均值差异。
- 第二阶段：如果 `lambda_inter_div` 证明有效，再评估 `selection_mode=diversity_first` 是否能让 checkpoint 选择更偏向非坍缩解。
- 第三阶段：只有当前两项有明确收益后，再加入 `action_aux`，判断动作监督能否进一步拉开动作间表征。
- 第四阶段：当抗坍缩最优组合稳定后，再回到 `pelvis_torso` 与 `mean_rms` 的语义一致性问题，确认不会重新引入坐标系回归。

## 新周期目标
- 目标一：把当前工作流重置为“先测试、再验证、后优化”的稳定循环，而不是并行混改。
- 目标二：维持或接近当前 `test_nMPJPE≈0.1943` 的精度水平，同时让预测分布显著摆脱平均姿态收缩。
- 目标三：让 `diagnose_pose_collapse.py` 中的 `variance_ratio_pred_over_target` 明显高于当前约 `0.07`，并使 `mse_pred_to_target` 稳定优于 `mse_meanpose_to_target`。
- 目标四：形成一套可以直接复用到后续模型、损失和归一化实验中的标准验证流程。
