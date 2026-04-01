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
- In progress: Phase 2 Step 2.2，将退回最保守的 `mean_rms` 标签归一化空间，筹备开展单环境（Env1）全量训练并观测是否成功打破平均坍缩。
- Pending: 重新引入 `action_aux` 动作监督或其他抗坍缩损失需要在网络有足够的表征容量以分离输入特征后再次测试。
- Pending: keep all validation and test execution aligned to the `WiFiPose` conda environment to avoid environment-dependent regressions.

## 测试与验证 Plan (Phase 1-4)
- **Phase 1：打通信息管道（核心：打破信息阻断，恢复过拟合能力）**
  - 目标：确保模型能通过死记硬背拟合训练集，观察 Train Loss 断崖式下降，Train nMPJPE 降至 0.05 以下。
  - Step 1.1：架构特征解封。在单环境下使用纯 MSE/SmoothL1 Loss，替换 ResNet1D 中的 `AdaptiveAvgPool1d(1)` 为 `Flatten()` 以保留空间分辨率。
  - Step 1.2：时序信息引入。如 1.1 失败，则切换至 `ms_tcn_pose` 并增加 `window_size` 以打破单帧歧义。
  - Step 1.3：单一 Batch 极限测试。关闭 Shuffle 用单 batch 过拟合 500 个 Epoch 排查数据纯度和梯度传导。
- **Phase 2：重塑数据与标签的“绝对一致性”（清洗燃料与标靶）**
  - 目标：消除人工预处理引入的歧义。在单环境（Env1）下使用最基础的 MSE Loss 进行测试。
  - Step 2.1：废除逐帧动态归一化，改为“全局/动作级归一化”。修改 `_normalize_aoa` 使用固定的物理阈值做 Min-Max 缩放，保留信号真实的能量强弱差异，让静止帧保持为接近 0 的低能量。
  - Step 2.2：退回最保守的标签空间（Mean-RMS）。坚决使用 `mean_rms`，彻底杜绝因为某些人躯干倾斜导致的“坐标尺度爆炸”。检验跑单环境全量数据，看 Train nMPJPE 能否跌破 0.10，Train std ratio 能否回升到 0.5 以上。
- **Phase 3：多任务辅助探针**
  - 目标：如果 Phase 2 之后依然有一定坍缩，用多任务来诊断和辅助网络打通宏观信号到二维坐标的映射。
  - Step 3.1：开启 Action Classification（动作分类）辅助分支。开启 `action_aux`。如果回归 Loss 下不去但分类 Accuracy 很高，说明输入特征有区分度但无法映射到细粒度坐标。分类 Loss 会强迫特征发散，打破平均坍缩。
- **Phase 4：引入物理先验与结构化约束（兜底阶段）**
  - 目标：恢复了全量数据上的过拟合能力后，为了提升测试集泛化性而做。
  - Step 4.1：加入结构化先验损失。打开 `lambda_dist` (骨长) 和 `lambda_rel` (角度) 等损失，用“人体骨架的物理常识”纠正 AoA 信号天生的一对多模糊性。
  - Step 4.2：极其谨慎地引入样本间排斥（Inter-Div Loss）。针对不同动作的样本开启 `lambda_inter_div` 强迫网络“遇到不同动作的标签时也必须把预测值拉开”。

## 新周期目标
- 目标一：通过打通信息管道彻底解决平均姿态塌缩的本因缺失。
- 目标二：分阶段、循序渐进地引入各种防塌损失设计与坐标归一，每一步均以是否出现断崖过拟合作为基线验证标准。
