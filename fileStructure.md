cpu-benchmark-analyzer/
│
├── data/                        # Data assets
│   ├── raw/                     # Unmodified source datasets
│   ├── processed/                # Cleaned datasets
│   └── features/                 # Engineered features
│
├── notebooks/                   # Prototyping & EDA
│   ├── 01_explore_data.ipynb
│   └── 02_model_prototype.ipynb
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── data/                    # Data pipeline (ETL)
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/                # Feature engineering
│   │   └── build_features.py
│   ├── models/                  # Training & inference
│   │   ├── train_model.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   ├── cli/                     # CLI interface
│   │   └── app.py               # Main CLI entrypoint
│   └── utils/                   # Helpers
│       └── io_utils.py
│
├── models/                      # Saved models
│   ├── checkpoints/
│   └── final_model.pkl
│
├── configs/                     # Config files
│   └── config.yaml              # Hyperparams, paths, etc.
│
├── tests/                       # Unit tests
│   └── test_preprocess.py
│
├── scripts/                     # Quick scripts
│   ├── run_etl.py               # Runs full ETL pipeline
│   ├── run_train.py             # Trains model
│   └── run_cli.sh               # Runs CLI for demo
│
├── requirements.txt             # Python dependencies
├── setup.py                     # Install as package (pip install -e .)
├── README.md
└── .gitignore
