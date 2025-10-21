import pandas as pd

def load_data(path: str):
    """
    Load dataset dynamically from the given path.
    """
    print(f"Loading dataset from: {path}")
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin1')

    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df