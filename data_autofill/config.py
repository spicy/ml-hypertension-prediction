from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@dataclass
class Config:
    """Configuration settings for data autofiller."""

    # Data Processing Settings
    SEQN_COLUMN: str = "SEQN"
    CHUNK_SIZE: int = 1000
    DEFAULT_ENCODING: str = "utf-8"

    # File Paths
    PROJECT_ROOT: Path = field(default_factory=get_project_root)
    DATA_DIR: Path = field(
        default_factory=lambda: get_project_root() / "data" / "processed"
    )
    PROCESSED_DIR: Path = field(
        default_factory=lambda: get_project_root() / "data" / "processed" / "autofilled"
    )
    QUESTIONS_DIR: Path = field(
        default_factory=lambda: get_project_root() / "questions"
    )

    # File Names
    AUTOFILLED_DATA_FILENAME: str = "AutofilledData.csv"

    # Processing Options
    ALLOW_MISSING_COLUMNS: bool = False
    ERROR_ON_MISSING_QUESTIONS: bool = True

    # Performance Settings
    PARALLEL_PROCESSING: bool = False
    MAX_WORKERS: Optional[int] = None


@dataclass
class LoggerConfig:
    """Configuration for logging settings."""

    LOGGER_NAME: str = "data_autofill"
    LOG_FORMAT: str = (
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s"
    )
    LOGS_FOLDER: str = "logs"
    LOG_FILE_NAME: str = "data_autofill.log"
    LOG_LEVEL: str = "DEBUG"
    LOG_COLORS: Dict[str, str] = field(
        default_factory=lambda: {
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        }
    )


config = Config()
logger_config = LoggerConfig()
