"""Evaluate a saved model on a dataset and print/save metrics."""
import argparse
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from src.features import sequence_to_counts
import numpy as np


def load_model(path):
    return joblib.load(path)


def featurize_with_alphabet(df, alphabet):
    X = np.vstack([sequence_to_counts(s, alphabet=alphabet) for s in df['sequence'].astype(str)])
    return X


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='results/metrics.json')
    args = p.parse_args()

    mobj = load_model(args.model)
    df = pd.read_csv(args.data)
    if 'property' in df.columns and 'target' not in df.columns:
        df = df.rename(columns={'property':'target'})
    X = featurize_with_alphabet(df, mobj.get('alphabet'))
    y = df['target'].values
    ypred = mobj['model'].predict(X)

    mse = mean_squared_error(y, ypred)
    r2 = r2_score(y, ypred)
    metrics = {'mse': float(mse), 'r2': float(r2)}
    import json
    with open(args.out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(metrics)

if __name__ == '__main__':
    main()
