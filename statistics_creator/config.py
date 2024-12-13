import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BaseConfig:
    """Base configuration class with common plot settings."""

    WIDTH: int = 15
    HEIGHT: int = 10
    TITLE_FONT_SIZE: int = 24
    TITLE_PAD: int = 20
    X_LABEL_FONT_SIZE: int = 20
    Y_LABEL_FONT_SIZE: int = 20
    LABEL_PAD: int = 15
    X_TICK_FONT_SIZE: int = 10
    Y_TICK_FONT_SIZE: int = 14
    TIGHT_LAYOUT_PAD: float = 8.0
    DPI: int = 300


@dataclass
class MissingDataConfig(BaseConfig):
    """Configuration for missing data visualization and analysis."""

    TEXT_FONT_SIZE: int = 5
    YLIM_MULTIPLIER: float = 1.15
    PLOT_TITLE: str = "Percentage of Missing Data by Column"
    X_LABEL: str = "Columns"
    Y_LABEL: str = "Percentage of Missing Data"
    PLOT_FILENAME: str = "missing_data_percentage.png"

    # New constants moved from MissingDataAnalyzer
    PERCENTAGE_MULTIPLIER: int = 100
    MISSING_DATA_LOG_MESSAGE: str = "Missing data analysis completed."


@dataclass
class CorrelationMulticollinearityConfig(BaseConfig):
    """Configuration for correlation and multicollinearity visualization and analysis."""

    CORR_WIDTH: int = 40
    CORR_HEIGHT: int = 30
    CORR_X_TICK_FONT_SIZE: int = 10
    CORR_Y_TICK_FONT_SIZE: int = 10
    CORR_TIGHT_LAYOUT_PAD: float = 3.0
    CORR_ANNOT_FONT_SIZE: int = 8

    VIF_WIDTH: int = 40
    VIF_HEIGHT: int = 30
    VIF_LABEL_FONT_SIZE: int = 16
    VIF_LEGEND_FONT_SIZE: int = 12
    VIF_TICK_FONT_SIZE: int = 10
    VIF_THRESHOLD: float = 5.0

    CORRELATION_PLOT_FILENAME: str = "correlation_matrix_heatmap.png"
    MULTICOLLINEARITY_PLOT_FILENAME: str = "multicollinearity_vif.png"

    IMPUTER_STRATEGY: str = "mean"
    VIF_SORT_ASCENDING: bool = False


@dataclass
class ClassDistributionConfig(BaseConfig):
    """Configuration for class distribution visualization and analysis."""

    WIDTH: int = 10
    HEIGHT: int = 6
    PIE_TEXT_FONT_SIZE: int = 12
    TITLE_PAD: int = 30
    TEXT_FONT_SIZE: int = 10
    PLOT_FILENAME: str = "class_analysis.png"
    SUBPLOT_LAYOUT: tuple = (1, 2)
    PIE_CHART_POSITION: tuple = (1, 1)
    INFO_TEXT_POSITION: tuple = (1, 2)
    PIE_CHART_AUTOPCT: str = "%1.1f%%"
    PIE_CHART_START_ANGLE: int = 90
    TEXT_BOX_POSITION: tuple = (0.5, 0.5)
    TEXT_BOX_ALIGNMENT: tuple = ("center", "center")
    TEXT_BOX_FACECOLOR: str = "white"
    TEXT_BOX_ALPHA: float = 0.5

    DISTRIBUTION_KEY: str = "distribution"
    MAJORITY_CLASS_KEY: str = "majority_class"
    MINORITY_CLASS_KEY: str = "minority_class"
    IMBALANCE_RATIO_KEY: str = "imbalance_ratio"


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@dataclass
class DataConfig:
    """Configuration for data file path and target column."""

    PROJECT_ROOT: Path = field(default_factory=get_project_root)
    DATA_DIR: Path = field(default_factory=lambda: get_project_root() / "data")
    AUTOFILLED_DIR: Path = field(
        default_factory=lambda: get_project_root() / "data" / "processed" / "autofilled"
    )
    FILTERED_DATA_PATTERN: str = "AutoFilled_Data_*.csv"
    TARGET_COLUMN: str = "HYPERTENSION"
    RESULTS_FILENAME: str = "analysis_results.json"
    DEFAULT_STATISTICS_FOLDER: str = (
        get_project_root() / "data" / "processed" / "autofilled" / "statistics"
    )


@dataclass
class FeatureImportanceConfig(BaseConfig):
    """Configuration for feature importance analysis."""

    # Visualization settings
    WIDTH: int = 12
    HEIGHT: int = 8
    LABEL_FONT_SIZE: int = 10
    TICK_FONT_SIZE: int = 8
    TIGHT_LAYOUT_PAD: float = 1.0

    # Plot filenames
    RF_IMPORTANCE_PLOT: str = "random_forest_importance.png"
    GB_IMPORTANCE_PLOT: str = "gradient_boosting_importance.png"
    F_SCORE_PLOT: str = "f_score_importance.png"

    # Analysis settings
    K_BEST_FEATURES: int = 20
    RANDOM_STATE: int = 42

    # Random Forest settings
    RF_N_ESTIMATORS: int = 700
    RF_MAX_DEPTH: int = 25
    RF_MIN_SAMPLES_SPLIT: int = 2
    RF_MIN_SAMPLES_LEAF: int = 8

    # Gradient Boosting settings
    GB_N_ESTIMATORS: int = 100
    GB_LEARNING_RATE: float = 0.05
    GB_MAX_DEPTH: int = 5
    GB_MIN_SAMPLES_LEAF: int = 1
    GB_MIN_SAMPLES_SPLIT: int = 10


@dataclass
class OutlierConfig(BaseConfig):
    """Configuration for outlier visualization and analysis."""

    WIDTH: int = 8
    HEIGHT: int = 5
    LABEL_FONT_SIZE: int = 10
    LEGEND_FONT_SIZE: int = 8
    TICK_FONT_SIZE: int = 6
    LOWER_BOUND_COLOR: str = "r"
    UPPER_BOUND_COLOR: str = "g"
    BOUND_LINESTYLE: str = "--"
    PLOT_FILE_PREFIX: str = "outliers_"
    PLOT_FILE_EXTENSION: str = ".png"

    # New constants moved from OutlierAnalyzer
    LOWER_QUANTILE: float = 0.25
    UPPER_QUANTILE: float = 0.75
    IQR_MULTIPLIER: float = 1.5


@dataclass
class ComprehensiveNumericalConfig(BaseConfig):
    """Configuration for comprehensive numerical analysis visualization and analysis."""

    WIDTH: int = 20
    HEIGHT: int = 14
    LABEL_FONT_SIZE: int = 12
    LEGEND_FONT_SIZE: int = 10
    HIST_BINS: int = 20
    HEATMAP_ANNOT_SIZE: int = 6
    BOXPLOT_WIDTH: float = 0.5
    DISTRIBUTION_PLOT_FILENAME: str = "numerical_distribution_{}.png"
    SUMMARY_HEATMAP_FILENAME: str = "summary_statistics_heatmap.png"

    NUMERIC_DTYPES = [np.number]
    SKEWNESS_KEY: str = "skewness"
    KURTOSIS_KEY: str = "kurtosis"


@dataclass
class LoggerConfig:
    """Configuration for logging settings."""

    LOGGER_NAME: str = "statistics_creator"
    LOG_FORMAT: str = (
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s"
    )
    LOGS_FOLDER: str = "logs"
    LOG_FILE_NAME: str = "statistics_creator.log"
    LOG_COLORS: dict = field(
        default_factory=lambda: {
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        }
    )


missing_data_config = MissingDataConfig()
correlation_multicollinearity_config = CorrelationMulticollinearityConfig()
class_distribution_config = ClassDistributionConfig()
data_config = DataConfig()
feature_importance_config = FeatureImportanceConfig()
outlier_config = OutlierConfig()
comprehensive_numerical_config = ComprehensiveNumericalConfig()
logger_config = LoggerConfig()
