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
- In progress: explain residual average-pose collapse with measurable evidence; although `nMPJPE` has recovered, `diagnose_pose_collapse.py` shows `variance_ratio_pred_over_target≈0.057` and the prediction distribution is still much narrower than the target distribution.
- In progress: use the recovered pure-accuracy baseline as the new control group and run single-factor anti-collapse experiments in a fixed order: `lambda_inter_div` first, then `selection_mode=diversity_first`, then `action_aux`.
- Pending: restore training and evaluation semantic consistency, especially whether `pelvis_torso` still carries regression risk relative to the recovered `mean_rms` baseline after anti-collapse terms are reintroduced.
- Pending: strengthen repeatable validation so changes affecting data, loss, or checkpoints are checked with `sanity_check/run_sanity_check.py`, `eval.py`, `diagnose_pose_collapse.py`, or `tools/diagnose_input_pose_separability.py`.
- Pending: keep all validation and test execution aligned to the `WiFiPose` conda environment to avoid environment-dependent regressions.

## 下阶段任务
- 第一阶段：将当前 `resnet1d + mean_rms + selection_mode=accuracy + zero diversity/action_aux` 结果固定为控制组，统一对照 `val_nMPJPE≈0.1948`、`test_nMPJPE≈0.1943`、`variance_ratio≈0.057`。
- 第二阶段：只加入 `lambda_inter_div`，其余设置保持不变，观察是否能在不显著破坏 `nMPJPE` 的前提下提升 `variance_ratio_pred_over_target`、`pred_group_std_mean` 与动作间均值差异。
- 第三阶段：只有当第二阶段证明 `lambda_inter_div` 有正收益后，才切换到 `selection_mode=diversity_first` 做单因素对照，判断 checkpoint 选择策略是否进一步改善跨动作分离。
- 第四阶段：只有当前两项都得到明确结论后，才加入 `action_aux`，评估动作监督是否能继续抬高预测分布多样性而不拉高回归误差。
- 第五阶段：在抗坍缩项的最优组合确定后，再回到 `pelvis_torso` 与 `mean_rms` 的语义一致性问题，确认新的最优配置不会重新引入坐标系回归。

## 下一阶段目标
- 目标一：确认当前问题已从“训练失败”收敛为“精度恢复但预测分布过窄”，后续优化重点转为提升跨样本与跨动作多样性。
- 目标二：在保持 `test_nMPJPE` 不明显差于 `0.1943` 的前提下，将 `variance_ratio_pred_over_target` 从约 `0.057` 明显抬升。
- 目标三：让 `diagnose_pose_collapse.py` 中的 `mse_pred_to_target` 稳定优于 `mse_meanpose_to_target`，避免模型在诊断上继续接近全局平均姿态基线。
- 目标四：形成一套固定、可复现的单因素抗坍缩实验顺序，避免再次出现“多项改动同时发生导致结论不可归因”的问题。
