from dataclasses import dataclass
import os

@dataclass
class BaseConfig:
    """Base configuration class with common plot settings."""
    WIDTH: int = 30
    HEIGHT: int = 24
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
    """Configuration for missing data visualization."""
    TEXT_FONT_SIZE: int = 5
    YLIM_MULTIPLIER: float = 1.15

@dataclass
class CorrelationConfig(BaseConfig):
    """Configuration for correlation visualization."""
    WIDTH: int = 40
    HEIGHT: int = 30
    X_TICK_FONT_SIZE: int = 10
    Y_TICK_FONT_SIZE: int = 10
    TIGHT_LAYOUT_PAD: float = 3.0
    ANNOT_FONT_SIZE: int = 8

@dataclass
class SummaryStatisticsConfig(BaseConfig):
    """Configuration for summary statistics visualization."""
    WIDTH: int = 60
    HEIGHT: int = 20
    X_TICK_FONT_SIZE: int = 10
    Y_TICK_FONT_SIZE: int = 14
    TIGHT_LAYOUT_PAD: float = 10.0

@dataclass
class ClassDistributionConfig(BaseConfig):
    """Configuration for class distribution visualization."""
    PIE_TEXT_FONT_SIZE: int = 24
    TITLE_PAD: int = 60
    TEXT_FONT_SIZE: int = 20

@dataclass
class DataConfig:
    """Configuration for data file path and target column."""
    PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '2017-2020', 'processed', 'FilteredCombinedData.csv')
    TARGET_COLUMN: str = "BPQ020"

@dataclass
class MulticollinearityConfig(BaseConfig):
    """Configuration for multicollinearity visualization."""
    WIDTH: int = 40
    HEIGHT: int = 30
    LABEL_FONT_SIZE: int = 16
    LEGEND_FONT_SIZE: int = 12
    TICK_FONT_SIZE: int = 10
    VIF_THRESHOLD: float = 5.0

@dataclass
class NumericalDistributionConfig(BaseConfig):
    """Configuration for numerical distribution visualization."""
    WIDTH: int = 40
    HEIGHT: int = 30
    LABEL_FONT_SIZE: int = 16
    LEGEND_FONT_SIZE: int = 12
    HIST_BINS: int = 30

@dataclass
class FeatureImportanceConfig(BaseConfig):
    """Configuration for feature importance visualization."""
    WIDTH: int = 12
    HEIGHT: int = 8
    LABEL_FONT_SIZE: int = 12
    TICK_FONT_SIZE: int = 10
    TIGHT_LAYOUT_PAD: float = 1.5

@dataclass
class OutlierConfig(BaseConfig):
    """Configuration for outlier visualization."""
    WIDTH: int = 40
    HEIGHT: int = 30
    LABEL_FONT_SIZE: int = 16
    LEGEND_FONT_SIZE: int = 12
    TICK_FONT_SIZE: int = 10

missing_data_config = MissingDataConfig()
correlation_config = CorrelationConfig()
summary_statistics_config = SummaryStatisticsConfig()
class_distribution_config = ClassDistributionConfig()
multicollinearity_config = MulticollinearityConfig()
numerical_distribution_config = NumericalDistributionConfig()
data_config = DataConfig()
feature_importance_config = FeatureImportanceConfig()
outlier_config = OutlierConfig()