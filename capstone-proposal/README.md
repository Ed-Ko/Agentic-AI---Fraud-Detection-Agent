# Financial Fraud Investigation with the Elliptic Bitcoin Dataset

This project investigates illicit Bitcoin transactions using the public Elliptic
transaction graph. It provides a validated data pipeline, two baseline models,
temporal evaluation, and reproducible tests.

## Current implementation

- Schema validation and dataset summaries
- Deterministic temporal train/validation/test splitting
- Rule-based threshold baseline
- Class-weighted XGBoost baseline
- Precision, recall, F1, PR-AUC, ROC-AUC, and confusion-matrix reporting

These are research baselines, not production fraud-detection decisions.

## Repository layout

```text
capstone-proposal/
├── README.md
├── PROJECT_SPECIFICATION.md
├── requirements.txt
├── Makefile
├── configs/config.yaml
├── data/{raw,sample,processed}/
├── docs/{dataset.md,workflow.md}
├── src/{data,baselines,evaluation,utils}/
├── experiments/       # Research notebooks
├── tests/             # Automated tests
└── results/{metrics,figures,predictions}/
```

## Dataset setup

Place these original files in `data/raw/`:

```text
elliptic_txs_features.csv
elliptic_txs_classes.csv
elliptic_txs_edgelist.csv
```

The loader expects 203,769 transactions, 234,355 directed edges, and 167
columns in the headerless feature file. See [docs/dataset.md](docs/dataset.md)
for the complete schema, published counts, and leakage considerations.

For a lightweight local fixture, create 100-row files in `data/sample/`:

```bash
python3 scripts/create_sample_dataset.py
python3 scripts/check_environment.py --data-dir data/sample
```

The fixture uses the same three filenames and 167-column feature schema, but
contains synthetic values and must not be used as research data.

The full downloaded dataset belongs in `data/raw/`; the small fixture belongs
in `data/sample/` and leaves the research data untouched.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Direct dependency versions are pinned for reproducibility.

## Run the workflow

From this directory:

```bash
make check   # Verify dependencies and all three CSV files
make build   # Train and evaluate both baselines once
make test    # Run automated tests
make all     # Run check, build, and test in order
```

The build uses a forward temporal split: train steps 1–34, validation steps
35–39, and test steps 40–49. Preprocessing and thresholds never use test labels.
Test metrics are written to `results/metrics/baseline_results.json`.

## Baselines

The rule model creates train-derived per-feature thresholds and selects its vote
cutoff on validation data. It reports whether validation accuracy falls within
the requested 70–80% range; unseen test accuracy cannot be guaranteed.

The XGBoost model uses deterministic settings, a fixed seed, and class weighting
for the illicit/licit imbalance. Its probability threshold is selected on
validation data before the final test evaluation.

## Research notebooks

The notebooks in `experiments/` cover the rule baseline, XGBoost baseline, and
baseline comparison. Reuse `src/baselines/run.py` so notebooks do not duplicate
split or preprocessing logic.

## Reproducibility and limitations

A temporal split is required because random splitting can expose future behavior
and inflate fraud-detection performance. Graph-derived neighborhood features
also require an explicit edge-visibility rule at scoring time. See
[docs/workflow.md](docs/workflow.md) for the complete process.
