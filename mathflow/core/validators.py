"""Input validation utilities."""

import numpy as np
import pandas as pd
from functools import wraps


def validate_matrix(data, name="matrix"):
    """Validate and convert input to numpy array."""
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.asarray(data, dtype=float)
    if data.ndim < 1:
        raise ValueError(f"{name} must be at least 1-dimensional, got {data.ndim}D")
    if np.any(np.isnan(data)):
        raise ValueError(f"{name} contains NaN values")
    return data


def validate_positive(data, name="data"):
    """Validate all values are positive."""
    data = validate_matrix(data, name)
    if np.any(data <= 0):
        raise ValueError(f"{name} must contain only positive values")
    return data


def validate_square_matrix(data, name="matrix"):
    """Validate input is a square matrix."""
    data = validate_matrix(data, name)
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {data.shape}")
    return data


def ensure_numpy(func):
    """Decorator to ensure DataFrame inputs are converted to numpy arrays."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                new_args.append(arg.values)
            elif isinstance(arg, pd.Series):
                new_args.append(arg.values)
            else:
                new_args.append(arg)
        return func(*new_args, **kwargs)

    return wrapper
