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


def validate_non_negative(data, name="data"):
    """Validate all values are non-negative."""
    data = validate_matrix(data, name)
    if np.any(data < 0):
        raise ValueError(f"{name} must contain only non-negative values")
    return data


def validate_square_matrix(data, name="matrix"):
    """Validate input is a square matrix."""
    data = validate_matrix(data, name)
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {data.shape}")
    return data


def validate_range(value, low=None, high=None, name="value"):
    """Validate value is within range [low, high]."""
    if low is not None and value < low:
        raise ValueError(f"{name} must be >= {low}, got {value}")
    if high is not None and value > high:
        raise ValueError(f"{name} must be <= {high}, got {value}")
    return value


def validate_callable(func, name="function"):
    """Validate that func is callable."""
    if not callable(func):
        raise TypeError(f"{name} must be callable, got {type(func).__name__}")
    return func


def validate_length_match(*arrays, names=None):
    """Validate that all arrays have the same length."""
    if len(arrays) < 2:
        return
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        if names:
            msg = f"Length mismatch: {', '.join(f'{n}={l}' for n, l in zip(names, lengths))}"
        else:
            msg = f"Length mismatch: {lengths}"
        raise ValueError(msg)


def validate_positive_integer(value, name="value"):
    """Validate value is a positive integer."""
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def validate_2d_array(data, name="data"):
    """Validate input is a 2D array."""
    data = validate_matrix(data, name)
    if data.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got {data.ndim}D")
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
