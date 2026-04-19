# WiFiPose - Project Summary (2026-04-19)

## 1. Project Overview
WiFiPose is a deep learning system designed to estimate 2D human poses (17 COCO keypoints) using WiFi signal characteristics, specifically Angle of Arrival (AoA) and Temporal Difference features. 

## 2. Core Architecture
- **Primary Model**: `ResNet1DPose` (A Residual 1D CNN paired with a Deep MLP head utilizing LayerNorm and Dropout).
- **Data Pipeline**: Driven by `AOASampleDataset`, backed by preprocessed AoA cache logic (`aoa_tof_estimation.py`).
- **Legacy Components**: Exploratory models (`ConvBaseline`, `MultiScaleTemporalPoseTCN`) have been comprehensively pruned to centralize the workflow around the ResNet architecture.

## 3. Current Status & Bottlenecks
- **Accomplishments**: The severe "average-pose collapse" issue has been resolved. Incorporating the Deep MLP head alongside temporal difference inputs effectively stabilized variance.
- **Active Bottleneck**: The primary challenge is now **cross-environment generalization**. Training quickly overfits to the source domain's multipath signatures (often hitting the best validation validation state at Epoch 1 or 2).
- **Official Baseline**: The standard non-DANN accuracy track (`linux_non_dann_accuracy_resnet1d_20260412-193627.log`) holds the current state-of-the-art result at **`test nMPJPE = 0.1900`**.
- **Active Experiments**: We are currently running the **Augmentation-first Matrix** (`short_aug`, `short_aug_mid`, `short_aug_strong`, etc.). The goal is to evaluate if runtime spatial/temporal augmentations combined with early stopping can break the `0.1900` barrier.

## 4. Diagnostics & Auxiliary Tracks
- **DANN (Domain Adversarial Neural Network)**: Kept alive specifically as a diagnostic feature to explicitly evaluate domain shifts, but it represents a secondary track since its baseline (`0.1902`) hasn't outperformed the standard track.
- **Alternative Checkpointing**: Selection paradigms like `balanced` and `diversity` are retained for analytical depth (improving `std_ratio`), but `accuracy` remains the strict overriding priority.

## 5. Workflow & Tooling
- **Windows (Local)**: Strictly restricted to development and smoke validation. Uses `tools/run_windows_smoke.py`. All local artifacts are intentionally sandboxed to `tmp/` to ensure no false formal metrics are logged.
- **Linux (Server)**: The exclusive authority for formal experiments. Uses `tools/run_linux_formal.py`, logging immutable tracked results to `logs/` and model weights to `checkpoints/`.

## 6. Recent Repository Cleanup
Conducted a medium-aggressiveness pruning on 2026-04-19:
- Removed obsolete models and dead code branches in `mymodels/` and `train.py`.
- Dropped deprecated tracking configs (`short_accuracy`, `short_reg_accuracy`) from the initial failed mitigation stage.
- Removed legacy validation scripts (`run_sanity_check.py`) and bash scheduling (`run_all.sh`) in favor of unified pure Python entry points.