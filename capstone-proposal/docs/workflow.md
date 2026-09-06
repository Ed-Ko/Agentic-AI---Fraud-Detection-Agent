# Baseline workflow

This process separates setup validation, one reproducible baseline experiment,
and testing. It never trains on the test period.

## 1. Initialize and check prerequisites

1. Create and activate a Python virtual environment if desired.
2. Install pinned-project minimum dependencies:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

3. Place the original three Elliptic CSV files in `data/raw/`.
4. Run `make check`. The preflight is read-only: it checks packages and files,
   and names every missing prerequisite.

## 2. Build models once

Run `make build`. It:

1. re-runs preflight validation;
2. loads and validates the Elliptic files;
3. splits labeled transactions into train (steps 1–34), validation (35–39),
   and test (40–49);
4. learns rule thresholds and XGBoost parameters on train only;
5. chooses the rule vote cutoff and XGBoost probability cutoff on validation
   only; and
6. prints and saves the untouched test metrics to
   `results/metrics/baseline_results.json`.

The JSON contains precision, recall, F1, PR-AUC, ROC-AUC, and a confusion
matrix for each model. The rule baseline also records whether its validation
accuracy fell within the 70–80% target; this is a target, not a test-set
guarantee.

## 3. Test

Run `make test` to execute `tests/`. Run `make all` to perform the complete
preflight → build → test sequence. A failed preflight stops the sequence before
any model is fit.
