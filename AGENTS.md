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
- Completed: 启动 Phase 1 Step 1.3，极大简化数据复杂度：关闭 Shuffle 用单一 Batch (batch_size=64) 极其暴力地迭代 500 次，以此来排查数据纯度和网络基础前传梯度是否完全正常。结果表明：在短短的 20 个 epochs 左右，模型 `train_loss` 迅速崩塌，并在 260 轮达到 `loss=0.000003` 且 `grad_norm=0.000`，实现了绝对的单样本记忆层级过拟合。这彻底证明了：**模型代码无 Bug、梯度传导完全正常、标签缩放范围合理。核心矛盾彻底落在了数据集的多样性映射上（同特征存在过多矛盾动作，或特征空间完全散乱），导致模型一旦引入跨 Batch 梯度立即发生坍缩拉平。**
- Completed: 更新 Phase 2 - Phase 4 最新研究计划，将核心矛盾锚定在 AOA 数据的预处理一致性和标签空间收敛上。
- Completed: 执行 Phase 2 Step 2.1，废除了 AOA 中的逐帧动态截断归一化，改为 `-25.0 ~ 0.0` 的全局固定物理阈值 Min-Max 缩放，真正保留了静态帧与剧烈动作帧在全局上的真实能量响应差异。
- Completed: 执行 Phase 2 Step 2.2，退回 `mean_rms` 标签归一化，跑单环境全量数据 (Env1)。测试表明即使在全局截断下，依然未能打破塌缩。Epoch 1 结束时 `train_loss=0.0224`，验证集 `val_nmpjpe=0.1989`，且 `train_std_ratio=0.1705` 依然深陷死水区（距离破 0.10、回升 0.5 的目标极远），这表明数据内部即使清理了归一化带来的额外畸变，其宏观映射依然存在深层的多对一交叉发散（同样的特征存在矛盾解）。
- Completed: Phase 2 未能完全恢复全量的过拟合，依据计划启动 Phase 3 多任务辅助分支（Action Auxiliary），强行用类别标签撕开特征空间分布。10 epochs 训练显示 `train_action_acc=72.5%`，`val_action_acc=3.4%`。
- Completed: 根据 Phase 3 实验结论，环境多径极度过拟合并且语义到坐标存在非线性映射断层，全面更新优化策略 Phase 1-3。
- Completed: Phase 1 Step 1.1 执行完毕，`aoa_dataset.py` 加入了与上一帧的数据作差的特征提取（Temporal Difference），以剥离静态反射。
- Pending: [Phase 1 Baseline] 执行帧间差分多环境联合基线测试（Train: Env1+Env2, Val: Env3），利用 `action_aux` 探针观察特征解耦能力，并记录基线 `std_ratio`。
- Pending: [Phase 2 Step 2.1] 升级非线性回归头（Deep MLP Head），加入 `LayerNorm` 和 `Dropout` 构建深层空间映射通道（输出17x2等价34维度）。
- Pending: [Phase 2 Evaluation] 保持超参数与随机种子 (`--seed 42`) 绝对一致重跑验证，核验升维网络是否将测验集预测方差比 (std_ratio) 大幅拔升至 >0.8 并彻底打破坍缩。

## 测试与验证 Plan (Phase 1-3 优化执行版)
- **Phase 1：物理特征解耦与跨域重建基线（解决 3.4% 跨域识别率）**
  - 测试标准：确立 Temporal Difference 后的纯净跨域表现，观察 action_aux 的 validation accuracy 提升情况，并以此划定坍缩评价准星。
- **Phase 2：回归头升维与非线性映射重构（解决 0.39 方差塌缩）**
  - **模块重构**：移除单薄的线性头，替换为 `Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm -> GELU -> Linear(34)` 深层映射网络，以跨越语义到坐标点阵的断层。
  - **绝地审判**：严格对齐基线测试的数据管线（同随机种子、同划分），以 `std_ratio` 是否断层拉升、以及测验集 `nMPJPE` 的真实表现为最终成果定论。
- **Phase 3：物理先验与多样性约束（兜底与精调）**
  - 目标：在前面打通“跨域特征”和“坐标映射”后，用 Loss 去约束不合理的人体比例。
  - Step 3.1：保持绝对稳健的归一化底座。保留 `normalize_mode: "mean_rms"` 与全局经验阈值截断。
  - Step 3.2：重新引入骨架物理约束。逐步将 `lambda_dist` 和 `lambda_rel` 调大。
  - Step 3.3：最后引入样本间差异惩罚（Inter-Div Loss）。开启拉大样本姿态距离的损失项 `lambda_inter_div`。

## 新周期目标
- 目标一：通过打通信息管道彻底解决平均姿态塌缩的本因缺失。
- 目标二：分阶段、循序渐进地引入各种防塌损失设计与坐标归一，每一步均以是否出现断崖过拟合作为基线验证标准。
