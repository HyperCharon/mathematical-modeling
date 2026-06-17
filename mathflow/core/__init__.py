"""Core utilities: configuration, logging, validation."""

from mathflow.core.config import Config, config
from mathflow.core.validators import (
    validate_matrix, validate_positive, validate_non_negative,
    validate_square_matrix, validate_range, validate_callable,
    validate_length_match, validate_positive_integer, validate_2d_array,
    ensure_numpy,
)
