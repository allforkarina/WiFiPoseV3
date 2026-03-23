# pose_estimationV2 项目总结与 2026-03-23 快速诊断

## 1. 项目目标

该项目的目标是根据 WiFi AoA 频谱估计人体二维姿态关键点。当前任务是一个监督式回归问题：

- 输入：单帧或短时间窗 AoA spectrum，长度为 `181`。
- 输出：`17 x 2` 的二维人体关键点坐标。
- 训练目标：在归一化姿态空间内最小化结构化姿态误差。
- 主要评估指标：`nMPJPE`。

当前工程已经具备完整训练链路：

1. 读取 AoA `.h5` 缓存和 `.npy` 姿态标签。
2. 做姿态归一化。
3. 送入 1D CNN 或时序 TCN 模型。
4. 用结构化损失训练。
5. 保存 checkpoint、history、曲线图和可视化评估结果。

## 2. 目录与职责

- `train.py`：训练主入口，负责数据划分、建模、训练、验证、测试、保存日志。
- `eval.py`：从 checkpoint 载入模型，按 action 抽样可视化预测结果。
- `dataloader/aoa_dataset.py`：AoA 数据和标签读取、姿态归一化、环境划分。
- `mymodels/conv1d_baseline.py`：轻量 1D CNN 基线。
- `mymodels/resnet1d_pose.py`：残差 1D CNN。
- `mymodels/temporal_tcn_pose.py`：多尺度时序 TCN，支持 `window_size > 1`。
- `dataloader/stratified_sampler.py`：分层 batch 采样。
- `diagnose_pose_collapse.py`：诊断预测是否塌缩到平均姿态。
- `sanity_check/run_sanity_check.py`：架构可训练性 sanity check。
- `configs/default.yaml`：默认训练、损失、模型、路径配置。

## 3. 数据与划分规则

根据当前代码和历史日志，真实数据不在仓库内，而是在外部路径：

- `D:\Files\WiFi_Pose\WiFiPoseV3\data\aoa_cache`
- `D:\Files\WiFi_Pose\WiFiPoseV3\data\dataset`

已确认数据规模为：

- 样本数：`1080`
- 总帧数：`320760`
- action 数：`27`
- 每个 sample 默认 `297` 帧

环境划分采用 sample id 推断：

- `S01-S10 -> env1`
- `S11-S20 -> env2`
- `S21-S30 -> env3`
- `S31-S40 -> env4`

训练默认使用：

- 验证集：`env3`
- 测试集：`env4`
- 训练集：其余环境

## 4. 数据流与建模方式

### 4.1 输入

`AOASampleDataset` 从每个 `.h5` 中读取 `aoa_spectrum`。单帧模型输入形状为：

- `(B, 1, 181)`，用于 `conv1d_baseline` 和 `resnet1d`

时序模型输入形状为：

- `(B, T, 181)`，例如 `window_size=5` 时送入 `ms_tcn_pose`

AoA 在进入模型前会进行：

- `nan/inf -> 0`
- 非负截断
- `log1p`
- 按 `p99` 和最大值归一化

### 4.2 标签

标签来自 `rgb/frame*.npy`，每帧为 `17 x 2` 关键点。

当前代码中的姿态归一化方式是：

- 以双髋中心作为平移中心
- 以髋中心到双肩中心的 torso 长度作为尺度
- 当 torso 长度过小，回退到 RMS 尺度

### 4.3 模型

- `ConvBaseline`：三层 1D 卷积 + 全局池化 + 线性头，结构最简单。
- `ResNet1DPose`：残差块堆叠，作为较稳的单帧基线。
- `MultiScaleTemporalPoseTCN`：先逐帧编码，再做多尺度时序卷积，使用中心时刻特征预测姿态。

## 5. 当前损失函数

`PoseStructureLoss` 当前由以下部分构成：

- `pose_loss`：Smooth L1 姿态回归
- `rel_loss`：关节两两相对位置误差
- `dist_loss`：关节两两距离误差
- `dir_loss`：居中后方向一致性
- `var_loss`：batch 级输出方差保持约束

`configs/default.yaml` 当前还新增了：

- `lambda_var: 0.1`
- `preserve_ratio: 0.6`

## 6. 历史有效结果

从已有日志看，项目曾经是能正常工作的，并且不同模型都达到过接近的效果：

- `default_resnet1d_20260323-051010_history.csv`
  - best val nMPJPE: `0.193553`
- `default_conv1d_baseline_20260323-045809_history.csv`
  - best val nMPJPE: `0.194323`
- `default_ms_tcn_pose_20260323-115115.log`
  - final val nMPJPE: `0.194816`
  - final test nMPJPE: `0.194003`

这说明：

- 数据本身不是不可学。
- 训练管线历史上可以收敛。
- 当前看到的明显变差，更像是“近期代码改动引入回归”。

## 7. 2026-03-23 本次快速短测

### 7.1 运行环境

- Python 环境：`D:\SoftWare\Anaconda\envs\WiFiPose\python.exe`
- PyTorch：`2.9.1+cu128`
- CUDA：可用

### 7.2 快速评估 1：历史 checkpoint 在当前代码下重新评估

命令：

```powershell
python eval.py --config configs/default.yaml --checkpoint checkpoints/resnet1d_env3val_env4test.pth --aoa_cache_root D:\Files\WiFi_Pose\WiFiPoseV3\data\aoa_cache --labels_root D:\Files\WiFi_Pose\WiFiPoseV3\data\dataset --model_name resnet1d --device cuda:0 --seed 42
```

结果：

- `resnet1d_env3val_env4test.pth`
  - `mean_nmpjpe = 0.464171`

命令：

```powershell
python eval.py --config configs/default.yaml --checkpoint checkpoints/ms_tcn_pose_relcoord_env3val_env4test.pth --aoa_cache_root D:\Files\WiFi_Pose\WiFiPoseV3\data\aoa_cache --labels_root D:\Files\WiFi_Pose\WiFiPoseV3\data\dataset --model_name ms_tcn_pose --window_size 5 --device cuda:0 --seed 42
```

结果：

- `ms_tcn_pose_relcoord_env3val_env4test.pth`
  - `mean_nmpjpe = 0.459338`

判断：

- 历史 checkpoint 在当前代码下重新评估时明显变差。
- 这与历史日志中的 `0.19x` 差距很大。
- 说明“当前数据归一化/评估语义”已经和历史 checkpoint 的训练语义不一致。

### 7.3 快速评估 2：当前代码的一步训练 smoke

命令：

```powershell
python train.py --config configs/default.yaml --aoa_cache_root D:\Files\WiFi_Pose\WiFiPoseV3\data\aoa_cache --labels_root D:\Files\WiFi_Pose\WiFiPoseV3\data\dataset --model_name ms_tcn_pose --window_size 5 --epochs 1 --max_steps 1 --checkpoint checkpoints\tmp_pose_collapse_smoke_20260323.pth
```

结果：

- train loss: `2.487962`
- val nMPJPE: `1.055544`
- test nMPJPE: `1.075427`
- val std ratio: `0.005046`
- test std ratio: `0.004022`

命令：

```powershell
python train.py --config configs/default.yaml --aoa_cache_root D:\Files\WiFi_Pose\WiFiPoseV3\data\aoa_cache --labels_root D:\Files\WiFi_Pose\WiFiPoseV3\data\dataset --model_name resnet1d --window_size 1 --epochs 1 --max_steps 1 --checkpoint checkpoints\tmp_resnet_smoke_20260323.pth
```

结果：

- train loss: `2.302533`
- val nMPJPE: `0.987241`
- test nMPJPE: `1.009391`
- val std ratio: `0.023347`
- test std ratio: `0.018602`

判断：

- 两个模型在当前代码下的一步 smoke 都表现很差。
- `std_ratio` 接近 `0`，尤其是 `ms_tcn_pose`，非常像“预测塌缩到近常数姿态”。
- 问题不是 `ms_tcn_pose` 独有，`resnet1d` 也受到明显影响。

## 8. 结论

当前项目的目标任务历史上是能正常执行的，但当前代码状态存在明显回归。

最有可能的原因不是数据坏掉，也不是 checkpoint 本身坏掉，而是最近两类修改改变了训练和评估的坐标系或优化行为：

1. `dataloader/aoa_dataset.py` 中姿态归一化从“全身均值 + RMS 尺度”改成了“骨盆中心 + torso 尺度”。
2. `train.py` / `configs/default.yaml` 中新增了 `variance preservation loss`。

当前现象与这两个改动高度一致：

- 历史 checkpoint 重评估变差，说明标签坐标系语义变了。
- 一步训练就出现极低 `std_ratio`，说明当前损失/归一化组合对模型产生了强塌缩风险。

## 9. 建议的修复顺序

建议按最小变量原则做 ablation，不要同时改多项。

### 方案 A：先恢复旧标签归一化，验证是否恢复历史表现

优先级最高。

- 把姿态归一化先回退到旧版本：
  - `center = pose.mean(axis=0, keepdims=True)`
  - `scale = sqrt(mean(sum(centered ** 2)))`
- 用历史 checkpoint 重新跑一次 `eval.py`
- 如果 `mean_nmpjpe` 回到约 `0.19x`，就说明主要回归源是归一化语义变化

### 方案 B：保留新归一化，但先禁用 `var_loss`

- 把 `lambda_var` 改为 `0.0`
- 重新做 `1 epoch / 1 step` smoke
- 观察：
  - `val/test std_ratio` 是否显著回升
  - `nMPJPE` 是否从 `1.0+` 回落

如果改善明显，再逐步尝试：

- `lambda_var = 0.01`
- `lambda_var = 0.05`
- `lambda_var = 0.1`

### 方案 C：固定 `resnet1d` 作为恢复基线

- 先只用 `resnet1d + window_size=1` 做回归恢复
- 因为它比时序模型更容易判断问题来自数据语义还是时序建模
- 等单帧基线恢复后，再恢复 `ms_tcn_pose`

### 方案 D：把历史可用配置固化成独立 config

- 新建一个基线配置文件，明确记录：
  - 旧归一化方式
  - 不带 `var_loss` 的损失设置
  - 稳定的模型和 `window_size`
- 避免以后“改了数据语义但沿用旧 checkpoint 名称和旧评估预期”

## 10. 当前已生成的短测产物

本次执行新增了以下运行产物：

- `checkpoints/tmp_pose_collapse_smoke_20260323.pth`
- `checkpoints/tmp_resnet_smoke_20260323.pth`
- `logs/train/default_ms_tcn_pose_20260323-230736.log`
- `logs/train/default_resnet1d_20260323-231332.log`
- `logs/eval/resnet1d_env3val_env4test/summary.csv`
- `logs/eval/ms_tcn_pose_relcoord_env3val_env4test/summary.csv`

这些结果已经足够支持当前结论，不建议在修复前继续长时间训练。
