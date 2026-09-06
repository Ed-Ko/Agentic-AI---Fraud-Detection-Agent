# Capstone Proposal

Starter structure for the capstone data-science project.

## Contents

- `PROJECT_SPECIFICATION.md` — project scope, requirements, and delivery plan.
- `configs/config.yaml` — shared configuration.
- `data/raw/` and `data/processed/` — source and transformed data.
- `src/` — data, baseline, evaluation, and shared utility code.
- `experiments/` — notebooks for rule, XGBoost, and comparison baselines.
- `tests/` — automated tests.
- `results/` — generated metrics, figures, and predictions.

## Getting started

1. Complete `PROJECT_SPECIFICATION.md` and update `configs/config.yaml`.
2. Add dependencies to `requirements.txt`.
3. Implement the data pipeline and baseline modules in `src/`.

## Workflow

Use the reproducible baseline workflow in [docs/workflow.md](docs/workflow.md):

```bash
make check  # validate packages and Elliptic CSV files
make build  # run each baseline once and save test metrics
make test   # run automated tests
```
