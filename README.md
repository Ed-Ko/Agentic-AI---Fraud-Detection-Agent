# Capstone Proposal Guideline

This repository contains a research-oriented capstone project for financial
fraud investigation using the Elliptic Bitcoin transaction dataset.

## Project structure

```text
capstone-proposal/
├── README.md                  # Project guide
├── PROJECT_SPECIFICATION.md   # Scope, requirements, and milestones
├── requirements.txt           # Pinned Python dependencies
├── Makefile                   # check, build, test, and all commands
├── configs/config.yaml        # Shared configuration
├── data/
│   ├── raw/                   # Original Elliptic CSV files
│   └── processed/             # Derived data
├── docs/
│   ├── dataset.md             # Schema, counts, leakage, and temporal split
│   └── workflow.md            # Reproducible process
├── scripts/
│   ├── check_environment.py   # Dependency/data preflight
│   ├── build_models.py        # One baseline run
│   └── run_tests.py           # Test-suite entry point
├── src/
│   ├── data/                  # Loading and preprocessing
│   ├── baselines/             # Rule and XGBoost models
│   ├── evaluation/            # Metrics and evaluation helpers
│   └── utils/                 # Shared utilities
├── experiments/               # Research notebooks
├── tests/                     # Automated tests
└── results/                   # Metrics, figures, and predictions
```

## Dataset requirements

Put the following files in `capstone-proposal/data/raw/`:

- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

The loader validates the published structure: 203,769 transactions, 234,355
directed edges, and 166 node attributes. See
[`capstone-proposal/docs/dataset.md`](capstone-proposal/docs/dataset.md) for
the file relationships, label counts, and leakage risks.

## Recommended workflow

From the `capstone-proposal/` directory:

```bash
python3 -m pip install -r requirements.txt
make check
make build
make test
```

Or run the complete sequence with:

```bash
make all
```

The baselines use a forward temporal split: training steps 1–34, validation
steps 35–39, and test steps 40–49. Transformations and thresholds are learned
without test labels. Metrics are saved in
`results/metrics/baseline_results.json`.

## Research expectations

Document the problem statement, users, goals, non-goals, technical approach,
risks, and acceptance criteria in `PROJECT_SPECIFICATION.md`. Use the supplied
loader and split utilities in experiments so results remain comparable and
reproducible. Report precision, recall, F1, PR-AUC, ROC-AUC, and the confusion
matrix for each baseline.
