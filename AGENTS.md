# Repository Guidelines

## Project Structure & Module Organization
`train.py` is the main training entry point. `eval.py` evaluates checkpoints and writes visualizations, and `diagnose_pose_collapse.py` compares saved models for collapse-related failure modes. Core model code lives in `mymodels/`, dataset loading and sampling live in `dataloader/`, preprocessing utilities live in `preprocess/`, and small shared helpers live in `utils/`. Runtime configuration is centralized in `configs/default.yaml`. Keep generated artifacts in `checkpoints/` and `logs/`; repository notes and progress records belong in `docs/`.

## Build, Test, and Development Commands
Use the repo from the project root with Python on your active environment.

```powershell
python train.py --config configs/default.yaml --verbose
python eval.py --config configs/default.yaml --checkpoint checkpoints/<model>.pth
python diagnose_pose_collapse.py --config configs/default.yaml
python sanity_check/run_sanity_check.py --epochs 5 --device cpu
python preprocess/aoa_tof_estimation.py --raw_data_root data/dataset --aoa_root data/aoa_cache
```

The first command trains a pose model, `eval.py` runs checkpoint evaluation and plotting, the diagnose script summarizes collapse metrics, and `run_sanity_check.py` verifies the forward/backward path on synthetic data.

## Coding Style & Naming Conventions
Follow the existing Python style: explicit imports, type hints where practical, and `pathlib.Path` for filesystem work. Use `snake_case` for functions, variables, and config keys; use `PascalCase` for classes. There is no committed formatter or linter config, so keep edits consistent with nearby files. Prefer 4-space indentation in new code, but preserve existing indentation in touched blocks to avoid noisy diffs.

## Testing Guidelines
There is no dedicated `tests/` package yet. Treat `sanity_check/run_sanity_check.py` as the minimum regression check before opening a PR, and run `eval.py` for model-affecting changes. For data or loss changes, include the exact command used, checkpoint path, and relevant metrics from `logs/` or generated plots.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `Add action-aware pair diversity loss` and `Use diversity-first checkpoint selection`. Keep commits focused and descriptive. PRs should state what changed, why it changed, which config values or data assumptions were affected, and how the change was validated. Include sample metrics or visualization outputs when behavior changes materially.

## Data & Configuration Notes
Prefer overriding dataset paths with CLI flags instead of hardcoding local directories. Do not commit raw datasets, cache files, checkpoints, or log outputs unless the change explicitly updates tracked documentation or small reference artifacts.
