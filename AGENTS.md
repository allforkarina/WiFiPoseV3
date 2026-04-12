# Repository Guidelines

## Project Structure & Module Organization
Core entry points are `train.py` for training and `eval.py` for checkpoint evaluation. Model definitions live in `mymodels/`, dataset and sampling code in `dataloader/`, preprocessing utilities in `preprocess/`, and shared helpers in `utils/`. Runtime configuration is centered in `configs/`, while longer notes and experiment summaries belong in `docs/`. Generated artifacts are typically written under `logs/`, `checkpoints/`, `results/`, and `tmp/`.

## Build, Test, and Development Commands
Run commands from the repository root.

- `python train.py --config configs/default.yaml` starts the main training loop with the default settings.
- `python eval.py --config configs/default.yaml --checkpoint checkpoints/<run>.pth` evaluates a saved checkpoint and writes visualizations under `logs/eval/`.
- `python sanity_check/run_sanity_check.py --epochs 5 --device cpu` runs a lightweight smoke test for forward, loss, backward, and optimizer flow.
- `python tools/prune_run_artifacts.py --keep 5` keeps only the newest training, evaluation, and checkpoint artifacts.
- `python preprocess/aoa_tof_estimation.py --help` shows preprocessing options for generating AoA caches.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, small focused functions, and `Path`-based filesystem handling where practical. Use `snake_case` for variables, functions, CLI flags, and config keys; use `PascalCase` for classes. Prefer explicit argument names in training and evaluation helpers. No formatter or linter configuration is checked in, so preserve surrounding style when editing.

## Testing Guidelines
This project relies on script-level validation rather than a formal `pytest` suite. Before submitting changes, run the sanity check and at least one task-relevant script such as `train.py` or `eval.py`. If you add tests, place them under `tests/` and use names like `test_<module>.py`.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit messages such as `Implement DANN architecture with dynamic alpha scheduling` and `add linux trainning config`. Keep subjects concise, action-first, and scoped to one change. Pull requests should describe behavior changes, note config or data-path updates, and include metrics, logs, or screenshots when training behavior changes.

## Configuration & Data Notes
Treat `configs/default.yaml` and `configs/linux.yaml` as the source of truth for paths, splits, and training defaults. Avoid hard-coding machine-specific paths in code; prefer configuration updates instead.
