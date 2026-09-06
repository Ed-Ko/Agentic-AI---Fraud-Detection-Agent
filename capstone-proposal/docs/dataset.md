# Elliptic Bitcoin dataset

## Expected files

Place the three original CSV files directly in `data/raw/` (or pass another
directory to `load_elliptic_dataset`):

| File | Role | Key columns |
| --- | --- | --- |
| `elliptic_txs_features.csv` | One row per Bitcoin transaction/node; it has no header. | column 0: `txId`; column 1: `time_step`; columns 2–166: anonymous numeric features |
| `elliptic_txs_classes.csv` | Labels for the transaction nodes. | `txId`, `class` |
| `elliptic_txs_edgelist.csv` | Directed Bitcoin payment-flow graph. | `txId1`, `txId2` |

`txId` is the join key.  Each feature row is a graph node.  A label row gives
that node `1` (illicit), `2` (licit), or `unknown`; labels are not a feature.
Each edge is directed from `txId1` to `txId2`, linking two nodes in the feature
table.  The loader verifies that class IDs and edge endpoints exist in that
table.

The feature CSV has 167 columns: an identifier plus 166 node attributes.  The
first attribute is the discrete `time_step` (1–49); the remaining 165
anonymous numeric attributes comprise 93 local transaction attributes and 72
one-hop aggregated-neighborhood attributes.  The released data does not expose
semantic feature names.  One time step represents approximately two weeks, so
the 49 steps span roughly two years.

## Published structure and counts

The original Elliptic dataset contains 203,769 transactions/nodes and 234,355
directed edges.  It has 4,545 labeled illicit transactions, 42,019 labeled
licit transactions, and 157,205 `unknown` transactions (46,564 labeled nodes
in total).  These are published reference counts; call `dataset_summary()`
after loading a local copy to record the exact counts and schema actually used
in an experiment.

The dataset description is based on the original [Elliptic research
paper](https://arxiv.org/abs/1908.02591) and the [released dataset
announcement](https://www.elliptic.co/newsroom/elliptic-releases-bitcoin-transactions-data/).

## Leakage risks

- **Randomly mixing time steps:** later transactions can reflect changed fraud
  patterns, later investigations, and information unavailable at an earlier
  decision time. A random split therefore produces an overly optimistic
  estimate for prospective fraud detection.
- **Fitting transformations globally:** imputers, scalers, encoders, feature
  selection, and resampling must be fitted only on training rows. The supplied
  `Standardizer` enforces this by separating `fit` and `transform`.
- **Graph leakage:** a graph model must not aggregate messages from future
  nodes or edges when scoring an earlier transaction. Use only edges and node
  features available at the score time; do not calculate global graph
  statistics across validation/test nodes before splitting.
- **Precomputed aggregated features:** the 72 neighborhood features may embed
  adjacent-transaction information. Treat their timestamp availability as an
  explicit assumption, audit it, and report results both with the complete
  feature set and (where appropriate) with local-only features.
- **Label-derived processing:** do not use labels from validation/test (or
  `unknown` nodes later adjudicated outside the release) in target encoding,
  threshold tuning, oversampling, feature selection, or graph label
  propagation.
- **Duplicate/linked transactions:** related transactions can span partitions.
  This is realistic for a temporal deployment but can still inflate results if
  graph features are built using future edges. Record the edge-visibility rule.

## Reproducible temporal protocol

A random train/test split is inappropriate for the primary evaluation because
`time_step` is chronological. Use a forward-chaining split on *labeled* nodes:

| Partition | Time steps | Purpose |
| --- | --- | --- |
| Train | 1–34 | fit preprocessing and model parameters |
| Validation | 35–39 | choose features, hyperparameters, and alert threshold |
| Test | 40–49 | one final, untouched prospective-style evaluation |

There are no edges across time steps: each period is a separate graph snapshot.
`temporal_split()` implements the 1–34/35–39/40–49 protocol, preserves input
order, excludes `unknown` labels by default, and rejects overlapping or
non-forward time ranges. Fix the split configuration in version control and
report class counts per partition. The temporal interpretation and snapshot
structure are described by [Elliptic's dataset
overview](https://medium.com/elliptic/the-elliptic-data-set-opening-up-machine-learning-on-the-blockchain-e0a343d99a14).
