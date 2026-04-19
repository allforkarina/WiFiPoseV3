# Repository Guidelines

## Project Structure & Module Organization
Core entry points are `train.py`, `eval.py`, and `diagnose_pose_collapse.py`. Model code lives in `mymodels/`, datasets in `dataloader/`, preprocessing in `preprocess/`, and shared utilities in `utils/`. Runtime configs live under `configs/`. Formal experiment logs are stored in `logs/`; temporary Windows smoke artifacts must stay under `tmp/`; checkpoints stay under `checkpoints/`.

## Build, Test, and Development Commands
Run commands from the repository root.

- `python train.py --config configs/default.yaml` runs the default Windows-facing training config.
- `python tools/run_windows_smoke.py --track non_dann` runs the fixed local non-DANN smoke path.
- `python tools/run_windows_smoke.py --track non_dann --variant short` runs the short-run early-overfit smoke path.
- `python tools/run_windows_smoke.py --track non_dann --variant mixed` runs the mixed-environment sequence-split smoke path for in-distribution overfit diagnosis.
- `python tools/run_windows_smoke.py --track non_dann --variant svd` runs the local SVD-residual-diff LOEO smoke path.
- `python tools/run_windows_smoke.py --track non_dann --variant mixed_svd` runs the local SVD-residual-diff mixed-split smoke path.
- `python train.py --config configs/windows_local_mixed_accuracy.yaml` runs the local Windows mixed-environment full-training diagnostic path when Linux is unavailable.
- `python tools/run_windows_smoke.py --track non_dann --variant short_reg` runs the stronger-regularization smoke path.
- `python tools/run_windows_smoke.py --track non_dann --variant short_reg_aug` runs the stronger-regularization plus augmentation smoke path.
- `python tools/run_windows_smoke.py --track non_dann --variant short_aug` runs the light-regularization plus augmentation smoke path.
- `python tools/run_windows_smoke.py --track non_dann --variant short_aug_mid` runs the medium-strength augmentation smoke path.
- `python tools/run_windows_smoke.py --track non_dann --variant short_aug_strong` runs the strong augmentation smoke path with light regularization.
- `python tools/run_windows_smoke.py --track non_dann --variant short_reg_aug_repro` reruns the current best regularization-plus-augmentation smoke path.
- `python tools/run_windows_smoke.py --track dann` runs the fixed local DANN smoke path.
- `python tools/run_linux_formal.py --track non_dann --variant accuracy` runs the formal non-DANN accuracy-first track.
- `python tools/run_linux_formal.py --track non_dann --variant mixed` runs the formal mixed-environment sequence-split non-DANN track for in-distribution overfit diagnosis.
- `python tools/run_linux_formal.py --track non_dann --variant svd` runs the formal LOEO SVD-residual-diff non-DANN track.
- `python tools/run_linux_formal.py --track non_dann --variant mixed_svd` runs the formal mixed-split SVD-residual-diff non-DANN track.
- `python tools/run_linux_formal.py --track non_dann --variant short` runs the short-run formal non-DANN track.
- `python tools/run_linux_formal.py --track non_dann --variant short_reg` runs the stronger-regularization formal non-DANN track.
- `python tools/run_linux_formal.py --track non_dann --variant short_reg_aug` runs the stronger-regularization plus augmentation formal non-DANN track.
- `python tools/run_linux_formal.py --track non_dann --variant short_aug` runs the light-regularization plus augmentation formal non-DANN track.
- `python tools/run_linux_formal.py --track non_dann --variant short_aug_mid` runs the medium-strength augmentation formal non-DANN track.
- `python tools/run_linux_formal.py --track non_dann --variant short_aug_strong` runs the strong augmentation formal non-DANN track with light regularization.
- `python tools/run_linux_formal.py --track non_dann --variant short_reg_aug_repro` reruns the current best regularization-plus-augmentation formal non-DANN track.
- `python tools/run_linux_formal.py --track non_dann --variant balanced` runs the formal non-DANN balanced-selection track.
- `python tools/run_linux_formal.py --track dann --variant accuracy` runs the formal DANN accuracy-first track.
- `python tools/run_linux_formal.py --track dann --variant balanced` runs the formal DANN balanced-selection track.
- `python eval.py --config configs/linux_non_dann.yaml --checkpoint checkpoints/<run>.pth` evaluates a saved checkpoint.
- `python tools/diagnose_aoa_env_diff.py --config configs/windows_smoke.yaml --compare_input_modes diff,svd_residual_diff` diagnoses and compares AoA feature modes without training.
- `python sanity_check/run_sanity_check.py --epochs 5 --device cpu` verifies the minimal forward/backward pipeline.
- `python tools/prune_run_artifacts.py --keep 5` trims old tracked logs and checkpoints.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, focused functions, explicit argument names, and `Path`-based filesystem handling where practical. Use `snake_case` for variables, functions, config keys, and CLI flags. Use `PascalCase` for classes. Preserve surrounding style when touching mixed-format legacy files.

## Testing Guidelines
This repository uses script-level validation instead of a formal `pytest` suite. On Windows, only run lightweight smoke validation: config load, dataset creation, one or two batches of forward/loss/backward, and checkpoint/log path checks. Do not run full experiments locally. Full training, evaluation, and official metrics must run on the Linux server after `git pull`.

## Commit & Pull Request Guidelines
Use short imperative commit messages scoped to one change, such as `Add smoke config and formal track split`. Every meaningful change must update this `AGENTS.md`, then be synced with `git add`, `git commit`, and `git push`. Include tracked `logs/` when the Linux server produces a formal result worth preserving.

## Configuration & Data Notes
`configs/default.yaml` remains the backward-compatible entry config. Preferred current configs are:

- `configs/windows_smoke.yaml` for local smoke-only validation
- `configs/windows_smoke_mixed.yaml` for local mixed-environment sequence-split smoke validation
- `configs/windows_smoke_svd.yaml` and `configs/windows_smoke_mixed_svd.yaml` for the first SVD-residual-diff smoke checks
- `configs/windows_local_mixed_accuracy.yaml` for local Windows mixed-environment full-training diagnosis
- `configs/windows_smoke_short_aug.yaml`, `configs/windows_smoke_short_aug_mid.yaml`, `configs/windows_smoke_short_aug_strong.yaml`, and `configs/windows_smoke_short_reg_aug_repro.yaml` for the augmentation-first smoke matrix
- `configs/windows_smoke_dann.yaml` for local DANN smoke validation
- `configs/linux_non_dann_accuracy.yaml`, `configs/linux_non_dann_balanced.yaml`, and `configs/linux_non_dann.yaml` for the non-DANN matrix
- `configs/linux_non_dann_mixed_accuracy.yaml` for the mixed-environment sequence-split non-DANN diagnostic track
- `configs/linux_non_dann_svd_accuracy.yaml` and `configs/linux_non_dann_mixed_svd_accuracy.yaml` for the first SVD-residual-diff formal checks
- `configs/linux_non_dann_short_accuracy.yaml`, `configs/linux_non_dann_short_reg_accuracy.yaml`, and `configs/linux_non_dann_short_reg_aug_accuracy.yaml` for the first non-DANN early-overfit mitigation matrix
- `configs/linux_non_dann_short_aml`, `configs/linux_dann_balanced.yaml`, and `configs/linux_dann.yaml` for the DANN matrix

Keep `domain_adaptation.use_dann` disabled unless the active experiment is explicitly the DANN track.
Keep `dataset.input_mode: "diff"` as the default baseline unless the active experiment explicitly targets feature engineering. The first feature-engineering candidate is `svd_residual_diff` with `dataset.svd_rank: 1`.

## Agent-Specific Instructions
- All user-facing replies, progress updates, and summaries must be written in Chinese.
- All content inside `AGENTS.md` must be written in English.
- Before any test or validation command, verify the runtime environment is `WiFiPose`; activate it first if needed.
- Windows is for code changes and smoke validation only. Linux is the only source of official full-training conclusions.
- After every meaningful code, config, or workflow change, update this file so the current optimization state remains handoff-ready.

## Current Optimization Targets
- **Completed**: Temporal Difference is integrated in the AoA dataset pipeline.
- **Completed**: Cleaned up deprecated legacy models (`ConvBaseline`, `TemporalTCN`), obsolete scripts, and old Phase 1 configurations to streamline the project architecture around `ResNet1DPose`.
- **Completed**: `ResNet1DPose` already uses the Deep MLP head (`Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm -> GELU -> Linear(34)`).
- **Completed**: The formal pre-DANN baseline has been run on Linux (`train20260412_2206.log`) with `mean_rms + action_aux + Deep MLP head + diversity_first`.
- **Completed**: The project is no longer primarily blocked by average-pose collapse. The active bottleneck is poor cross-environment generalization.
- **Completed**: The workflow is now split into a Windows smoke path (`configs/windows_smoke.yaml`, `tools/run_windows_smoke.py`) and explicit Linux formal tracks (`configs/linux_non_dann.yaml`, `configs/linux_dann.yaml`, `tools/run_linux_formal.py`).
- **Completed**: Track A code support is in place. The non-DANN line now has explicit `accuracy`, `balanced`, and `diversity` checkpoint-selection variants.
- **Completed**: Track B code support is in place. The DANN line now mirrors the same `accuracy`, `balanced`, and `diversity` checkpoint-selection variants, and Windows can smoke-test the DANN path locally.
- **Completed**: `eval.py` has been re-aligned with the current `train.py` interfaces so the formal evaluation path is runnable again.
- **Completed**: The six-run Linux comparison matrix has been pulled back into `logs/train` and reviewed on Windows.
- **Completed**: `accuracy` is the only checkpoint-selection rule that remains competitive on final test `nMPJPE`. `balanced` and `diversity` improve `std_ratio` but consistently worsen pose accuracy.
- **Completed**: The matched `accuracy` comparison shows no meaningful DANN gain under the current setup. `linux_non_dann_accuracy_resnet1d_20260412-193627.log` remains the official best run with final test `nMPJPE=0.1900`, while `linux_dann_accuracy_resnet1d_20260413-031611.log` finishes at `0.1902`.
- **Completed**: Across the reviewed matrices, the best validation `nMPJPE` appears at epoch 1, which indicates the current training recipe overfits cross-environment generalization almost immediately.
- **Completed**: `non-DANN + accuracy` is locked as the official baseline and remains the default comparison target for future ablations.
- **Completed**: The training loop now supports opt-in epoch-level LR scheduling, opt-in early stopping, runtime train-only AoA augmentation, and explicit overfit diagnostics (`lr`, `best_val_epoch`, `epochs_since_best`, `val_gap_from_best`, final `[overfit]` summary).
- **Completed**: The first four-run non-DANN early-overfit mitigation matrix (`baseline`, `short`, `short_reg`, `short_reg_aug`) has been reviewed. `short_reg_aug` nearly matches the official baseline at `test nMPJPE=0.1901` and is the healthiest overfit-mitigation candidate so far.
- **Completed**: The first mitigation review shows that stronger regularization alone does not help. The useful signal comes from the short-schedule plus runtime-augmentation combination.
- **Completed**: Best-checkpoint logging and best-so-far diagnostics now use the same checkpoint-improvement rule, while early stopping keeps its own `min_delta`-based patience anchor.
- **Completed**: A mixed-environment sequence-level split protocol is now available for in-distribution overfit diagnosis. It keeps each `(action, sample)` sequence intact and splits each `(action, env)` bucket with a fixed `7:2:1` train/val/test ratio.
- **Completed**: A local Windows full-training mixed split config is available for temporary server-maintenance periods. It uses `cuda:0`, `batch_size=32`, and early stopping for in-distribution overfit diagnosis only.
- **Pending / Next Stage**: Run the augmentation-first four-run Linux matrix: `short_aug`, `short_aug_mid`, `short_aug_strong`, `short_reg_aug_repro`.
- **Pending / Next Stage**: Use the augmentation-first matrix to decide whether runtime augmentation alone can beat `0.1900`, or whether the stronger `short_reg_aug` recipe needs to remain the leading candidate.
- **Completed**: The AoA input pipeline now supports configurable feature modes at the dataset layer: `diff`, `abs`, `svd_residual`, and `svd_residual_diff`.
- **Completed**: The feature-diagnosis tool can now compare multiple `input_mode` values in one run and report the best environment-gap and action-separation candidate.
- **Pending / Diagnostic**: Compare `diff` vs `svd_residual_diff` on Linux with `tools/diagnose_aoa_env_diff.py --compare_input_modes diff,svd_residual_diff`, then decide whether the SVD path should enter formal mixed training.
- **Pending / Next Stage**: If the diagnosis favors `svd_residual_diff`, run `mixed_svd` first and only promote the SVD path to LOEO formal training if the mixed split shows healthier overfit behavior.
- **Pending / Diagnostic**: Verify why `linux_dann_diversity_resnet1d_20260413-083219.log` currently ends at epoch 64 without final `train/test eval` lines before treating that run as a formal completed result.

## Testing and Validation Plan
- Windows validation must use the `windows_smoke*.yaml` configs and should only touch temporary outputs under `tmp/windows_smoke/`.
- Temporary local full-training diagnostics on Windows must use `configs/windows_local_mixed_accuracy.yaml` and keep artifacts under `tmp/windows_local_train/`.
- Windows-only AoA diagnosis runs must use `tools/diagnose_aoa_env_diff.py` and keep outputs under `tmp/diagnostics/aoa_env_diff/`.
- Linux AoA diagnosis runs may write to `logs/diagnose/aoa_env_diff/` so the CSV and PNG outputs can be pulled back with `scp`.
- Every Linux formal run must produce tracked `logs/` and be compared with the current official baseline `linux_non_dann_accuracy_resnet1d_20260412-193627.log` (`test nMPJPE=0.1900`).
- The mixed-environment sequence-split track is diagnostic only. It answers in-distribution overfit questions and must not replace the official cross-environment LOEO baseline.
- The default formal checkpoint-selection rule for official comparisons is now `accuracy`.
- `balanced` and `diversity` may still be run as diagnostics, but they must not be treated as primary selection rules unless they also improve final `nMPJPE`.
- The primary acceptance metric is always `val/test nMPJPE`.
- `std_ratio`, `action_aux` validation accuracy, and domain-classifier metrics are secondary diagnostics and must not override worse pose accuracy.
- The first feature-engineering gate is now `mixed_svd`: only if the SVD-residual-diff path improves overfit health there should it be promoted to the formal LOEO comparison.
- For the current overfit-mitigation stage, every formal log should also report `best_val_epoch`, `first_degradation_epoch`, `final_val_gap`, and whether early stopping triggered, and these fields must stay consistent with any saved best checkpoint.
- The current augmentation-first acceptance target is to beat the official baseline `0.1900`, or at minimum beat `short_reg_aug` (`test nMPJPE=0.1901`) while keeping `best_val_epoch >= 2` and a smaller `final_val_gap`.
- Any future DANN formal run must be considered incomplete if the log does not include final `train eval`, `test eval`, and assessment lines.

## Current Workflow
1. Update code or configs on Windows.
2. Run `python tools/run_windows_smoke.py --track <non_dann|dann> [--variant <baseline|mixed|svd|mixed_svd|short|short_reg|short_reg_aug|short_aug|short_aug_mid|short_aug_strong|short_reg_aug_repro>]` inside `WiFiPose`.
3. When feature-level diagnosis is needed, run `python tools/diagnose_aoa_env_diff.py --config <windows_or_linux_config> --compare_input_modes diff,svd_residual_diff` and store Windows outputs under `tmp/diagnostics/aoa_env_diff/` or Linux outputs under `logs/diagnose/aoa_env_diff/`.
4. Update `AGENTS.md`, commit, and push.
5. Pull on Linux and run the chosen formal track. For the SVD path, always run `mixed_svd` before `svd`; for the current augmentation-first baseline path, `short_aug`, `short_aug_mid`, `short_aug_strong`, and `short_reg_aug_repro` remain the active comparison matrix.
6. Push formal `logs/` back to GitHub.
7. Pull the new logs on Windows, compare them against the `0.1900` official baseline and the `0.1901` `short_reg_aug` candidate, and update the next optimization target list.
