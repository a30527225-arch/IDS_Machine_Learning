# utils/dummy_simulator.py
import numpy as np
import pandas as pd

def generate_random_traffic(feature_names, scaler, n_samples=10):
    """
    Generate dummy network traffic using the scaler's training statistics.
    This simulates realistic random packets.
    """
    if hasattr(scaler, 'mean_'):
        means = scaler.mean_
        stds = np.sqrt(scaler.var_) if hasattr(scaler, 'var_') else np.ones_like(means)
    else:
        means = np.zeros(len(feature_names))
        stds = np.ones(len(feature_names))
    data = np.random.normal(loc=means, scale=stds, size=(n_samples, len(feature_names)))
    return pd.DataFrame(data, columns=feature_names)

def generate_random_input_dict(features, scaler):
    """Return a single random sample as a dictionary for manual input."""
    df = generate_random_traffic(features, scaler, 1)
    return df.iloc[0].to_dict()