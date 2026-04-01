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
- Completed: 重新建立干净的控制组实验，只使用 `resnet1d + mean_rms + selection_mode=accuracy + zero diversity/action_aux`。
- Completed: 引入跨动作多样性损失 `lambda_inter_div=0.25` 开展单一变量控制实验，实验表明此损失将模型测试验证集指标表现降低至 `test_nMPJPE=0.2023`，但使得预测分布方差比 `variance_ratio_pred_over_target` 从 `0.07` 提升到 `0.1270`！
- Completed: 下一步实验测试 `selection_mode=diversity_first` 是否能让 checkpoint 选择更偏向非坍缩解。实验表明此策略成功将最佳 Checkpoint 的 `variance_ratio` 进一步拔高到了惊人的 `0.6779`，但付出了精度受损的代价（`test_nMPJPE` 退化至 `0.2557`，且 `mse_pred_to_target` 达到了 `0.0539`）。这说明模型产出了差异极大的动作，但部分动作未能对齐真实标签。
- Completed: 暂时搁置附加损失和多样性优化，为了定位“网络是否能够拟合训练集”以及“坍缩是否由于结构瓶颈导致”，完成了一次“纯净版 Vanilla”实验（关闭一切附加约束损失、多样性损失，关闭 Dropout 等，仅仅使用 Huber Loss）。测试结果表明：`train_nMPJPE=0.1972`，`test_nMPJPE=0.1950`，且 `mse_pred=0.0382` 甚至差于 `mse_meanpose=0.0373`。这证明在使用 ResNet1D（内置 `AdaptiveAvgPool1d`）以及当前 AoA 特征的数据结构下，网络连**训练集本身都无法真正过拟合**，其对空间特征的理解完全退化成了输出平均姿态。这指向了特征预处理机制和网络池化层的深层结构瓶颈！
- In progress: 重构或梳理网络结构与特征提取逻辑，尤其是 `AdaptiveAvgPool1d` 的空间维度坍缩问题。
- Pending: 重新引入 `action_aux` 动作监督或其他抗坍缩损失需要在网络有足够的表征容量以分离输入特征后再次测试。
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
