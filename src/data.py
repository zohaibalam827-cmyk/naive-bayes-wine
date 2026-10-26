"""Load the Wine dataset (saved as a CSV in data/)."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_wine

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "wine.csv"
LABEL = "cultivar"


def save_wine_csv(path=DATA_PATH):
    """Write the bundled Wine dataset to a CSV file."""
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df[LABEL] = [wine.target_names[t] for t in wine.target]
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_data(path=DATA_PATH):
    """Read the CSV, creating it first if it doesn't exist.

    Returns the feature matrix X, the labels y and the feature names."""
    if not path.exists():
        save_wine_csv(path)
    df = pd.read_csv(path)
    features = [c for c in df.columns if c != LABEL]
    return df[features].values, df[LABEL].values, features
