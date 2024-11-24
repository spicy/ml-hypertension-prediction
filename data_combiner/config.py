import os
from dataclasses import dataclass, field
from pathlib import Path

import colorlog


def get_project_root() -> Path:
    """Get the project root directory."""
    # Assuming data_combiner is one level below project root
    return Path(__file__).parent.parent


@dataclass
class Config:
    """Configuration settings for data combiner."""

    SEQN_COLUMN: str = "SEQN"
    PROJECT_ROOT: Path = field(default_factory=get_project_root)
    DATA_DIR: Path = field(default_factory=lambda: get_project_root() / "data")
    RAW_DATA_DIR: Path = field(
        default_factory=lambda: get_project_root() / "data" / "raw"
    )
    PROCESSED_DIR: Path = field(
        default_factory=lambda: get_project_root() / "data" / "processed"
    )
    QUESTIONS_DIR: Path = field(
        default_factory=lambda: get_project_root() / "questions"
    )
    UNFILTERED_DATA_FILENAME: str = "UnfilteredCombinedData.csv"
    FILTERED_DATA_FILENAME: str = "FilteredCombinedData.csv"
    VALIDATE_DATA: bool = False
    MIN_AGE: int = 18


@dataclass
class LoggerConfig:
    """Configuration for logging settings."""

    LOGGER_NAME: str = "data_combiner"
    LOG_FORMAT: str = (
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s"
    )
    LOGS_FOLDER: str = "logs"
    LOG_FILE_NAME: str = "data_combiner.log"
    LOG_COLORS: dict = field(
        default_factory=lambda: {
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        }
    )
    LOG_LEVEL: str = "DEBUG"


config = Config()
logger_config = LoggerConfig()
