"""Global configuration for MathFlow."""

from dataclasses import dataclass


@dataclass
class Config:
    """Global configuration."""

    # Visualization
    figure_dpi: int = 150
    figure_style: str = "seaborn-v0_8-whitegrid"
    font_family: str = "SimHei"  # Chinese font support
    font_size: int = 12

    # Numerical precision
    decimal_places: int = 4
    epsilon: float = 1e-10

    # AHP consistency ratio threshold
    ahp_cr_threshold: float = 0.10

    # Random seed for reproducibility
    random_seed: int = 42

    # Output
    save_format: str = "png"
    save_dpi: int = 300


# Global singleton
config = Config()
