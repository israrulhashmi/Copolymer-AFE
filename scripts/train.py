"""Train a baseline XGBoost or RandomForest model on sequence -> property data."""
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.features import sequence_to_counts
import numpy as np


def load_data(path):
    df = pd.read_csv(path)
    # expect columns: sequence, target
    if 'sequence' not in df.columns:
        raise ValueError('CSV must contain a "sequence" column')
    if 'target' not in df.columns and 'property' in df.columns:
        df = df.rename(columns={'property':'target'})
    if 'target' not in df.columns:
        raise ValueError('CSV must contain a "target" or "property" column')
    return df


def featurize(df):
    # naive fixed alphabet from dataset
    allseq = ''.join(df['sequence'].astype(str).tolist())
    alphabet = sorted(set(allseq))
    X = np.vstack([sequence_to_counts(s, alphabet=alphabet) for s in df['sequence'].astype(str)])
    return X, df['target'].values, alphabet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='path to CSV data')
    p.add_argument('--model', choices=['xgboost','rf'], default='xgboost')
    p.add_argument('--output', default='models/model.joblib')
    args = p.parse_args()

    df = load_data(args.data)
    X, y, alphabet = featurize(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if args.model == 'xgboost':
        m = XGBRegressor(n_estimators=100, random_state=42)
    else:
        m = RandomForestRegressor(n_estimators=100, random_state=42)

    m.fit(X_train, y_train)

    joblib.dump({'model': m, 'alphabet': alphabet}, args.output)
    print(f'Saved model to {args.output}')

if __name__ == '__main__':
    main()
