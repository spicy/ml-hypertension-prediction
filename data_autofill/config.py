from dataclasses import dataclass, field
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@dataclass
class Config:
    """Configuration settings for data autofiller."""

    SEQN_COLUMN: str = "SEQN"
    PROJECT_ROOT: Path = field(default_factory=get_project_root)
    DATA_DIR: Path = field(default_factory=lambda: get_project_root() / "data")
    PROCESSED_DIR: Path = field(
        default_factory=lambda: get_project_root() / "data" / "processed"
    )
    QUESTIONS_DIR: Path = field(
        default_factory=lambda: get_project_root() / "questions"
    )
    AUTOFILLED_DATA_FILENAME: str = "AutofilledData.csv"


@dataclass
class LoggerConfig:
    """Configuration for logging settings."""

    LOGGER_NAME: str = "data_autofill"
    LOG_FORMAT: str = (
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s"
    )
    LOGS_FOLDER: str = "logs"
    LOG_FILE_NAME: str = "data_autofill.log"
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
