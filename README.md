Capstone proposal guideline

File Structure
capstone-proposal/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessing.py
│   │
│   ├── baselines/
│   │   ├── rules.py
│   │   └── xgboost.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate.py
│   │
│   └── utils/
│       └── ...
│
├── experiments/
│   ├── 01_rule_baseline.ipynb
│   ├── 02_xgboost_baseline.ipynb
│   └── 03_baseline_comparison.ipynb
│
├── tests/
│
└── results/
    ├── metrics/
    ├── figures/
    └── predictions/