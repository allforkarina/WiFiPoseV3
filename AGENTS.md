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
- Completed: 移除 ResNet1D 中的 `AdaptiveAvgPool1d(1)` 并使用 `Flatten()` 进行单一环境 (env1) 过拟合实验 (Phase 1 Step 1.1)。结果发现 Train Loss 仍未出现断崖式下降，停留在 0.015-0.02的水平。证明仍然未能过拟合。
- Completed: 启动 Phase 1 Step 1.2，引入 `ms_tcn_pose` 并增加 `window_size=11` 来引入时序信息，测试能否打破单帧歧义并实现单环境过拟合。结果表明：到了第 15 个 epoch 时，`train_loss` 仅降低至 0.0062（对应 nMPJPE 仍高于 0.05 期望值，并且验证集 `val_nmpjpe` 恶化维持在 0.22 左右），说明引入时间窗依然不能达到“死记硬背”级过拟合，模型特征提取能力由于特征本身的质量或纯净度仍然存在根本性阻断！
- In progress: 启动 Phase 1 Step 1.3，极大简化数据复杂度：关闭 Shuffle 用单一 Batch (batch_size=64) 极其暴力地迭代 500 次，以此来排查数据纯度和网络基础前传梯度是否完全正常。
- Pending: 重新引入 `action_aux` 动作监督或其他抗坍缩损失需要在网络有足够的表征容量以分离输入特征后再次测试。
- Pending: keep all validation and test execution aligned to the `WiFiPose` conda environment to avoid environment-dependent regressions.

## 测试与验证 Plan (Phase 1-4)
- **Phase 1：打通信息管道（核心：打破信息阻断，恢复过拟合能力）**
  - 目标：确保模型能通过死记硬背拟合训练集，观察 Train Loss 断崖式下降，Train nMPJPE 降至 0.05 以下。
  - Step 1.1：架构特征解封。在单环境下使用纯 MSE/SmoothL1 Loss，替换 ResNet1D 中的 `AdaptiveAvgPool1d(1)` 为 `Flatten()` 以保留空间分辨率。
  - Step 1.2：时序信息引入。如 1.1 失败，则切换至 `ms_tcn_pose` 并增加 `window_size` 以打破单帧歧义。
  - Step 1.3：单一 Batch 极限测试。关闭 Shuffle 用单 batch 过拟合 500 个 Epoch 排查数据纯度和梯度传导。
- **Phase 2：重塑数据与标签空间**
  - 目标：特征清晰，标签稳健。
  - Step 2.1：退回最稳健的标签归一化。放弃 `pelvis_torso`，退回稳定的 `mean_rms` 模式。
  - Step 2.2：检查输入 AoA 的动态截断。探索 `_normalize_aoa` 从逐帧 Min-Max 改为全局固定最大最小值。
- **Phase 3：基础结构化约束的谨慎回归**
  - 目标：确认极强拟合能力后，让其“长得更像人”。
  - Step 3.1：加入骨长与角度损失。Test 出现退化后，引入 `lambda_dist` 和 `lambda_rel` 损失。
  - Step 3.2：防塌缩正则化。极缓慢引入多样本差异约束 `lambda_inter_div` 预防微小动作模糊。
- **Phase 4：跨域与泛化验证**
  - 目标：解决真实的跨环境问题。
  - Step 4.1：多环境联合训练 (env1~3)。
  - Step 4.2：跨环境 Zero-Shot 测试 (env4)，严重掉点则考虑域适应（DANN）。

## 新周期目标
- 目标一：通过打通信息管道彻底解决平均姿态塌缩的本因缺失。
- 目标二：分阶段、循序渐进地引入各种防塌损失设计与坐标归一，每一步均以是否出现断崖过拟合作为基线验证标准。
