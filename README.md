# Copolymer AFE Prediction (Sequence → Property)

This repository contains a **reproducible ML pipeline** to predict **Adsorption Free Energy (AFE)** (regression target) from **copolymer sequence descriptors** (input features). It includes baseline and production-ready training/evaluation scripts for:

- **XGBoost** (gradient-boosted trees)
- **Random Forest**
- **DNN (TensorFlow/Keras)**

The code is organized so you can:
1) drop in your `sequence.txt` and `property.txt` (or a CSV),  
2) train/evaluate multiple models with consistent splits and metrics, and  
3) generate parity plots and summary tables.

> Data files are **not committed** by default. See `data/README.md` for expected formats.

---

## Quickstart

### 1) Create environment (conda)
```bash
conda env create -f environment.yml
conda activate copolymer-afe
```

### 2) Add your data
Place your dataset in `data/` as described in `data/README.md`. If you have a single CSV file, name it `data/dataset.csv`.

### 3) Train a baseline model (XGBoost)
```bash
python scripts/train.py --data data/dataset.csv --model xgboost --output models/xgb.joblib
```

### 4) Evaluate
```bash
python scripts/evaluate.py --model models/xgb.joblib --data data/dataset.csv --out results/metrics.json
```

## Project layout
```
- src/          # python package: feature builders, data loaders
- scripts/      # training and evaluation scripts
- data/         # place datasets here (not committed)
- models/       # saved models
- results/      # figures and metrics
- notebooks/    # exploratory notebooks
- configs/      # experiment configs
```

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contact
Repo owner: @israrulhashmi
