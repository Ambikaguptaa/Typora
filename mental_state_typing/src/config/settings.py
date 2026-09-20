"""Application settings and environment configuration.

Loads configuration from .env file or environment variables with sensible defaults.
Designed to be modular, simple, and self-contained.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory points to the root of mental_state_typing/
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env file if present
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


@dataclass(frozen=True)
class Settings:
    """Application settings dataclass."""

    # Environment
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Project directories
    base_dir: Path = BASE_DIR
    raw_data_path: Path = Path(os.getenv("RAW_DATA_PATH", BASE_DIR / "data" / "raw"))
    processed_data_path: Path = Path(
        os.getenv("PROCESSED_DATA_PATH", BASE_DIR / "data" / "processed")
    )
    sample_data_path: Path = Path(
        os.getenv("SAMPLE_DATA_PATH", BASE_DIR / "data" / "sample")
    )
    models_path: Path = Path(os.getenv("MODELS_PATH", BASE_DIR / "models"))

    # Database configuration
    database_path: Path = Path(
        os.getenv("DATABASE_PATH", BASE_DIR / "database" / "mental_state.db")
    )

    # Privacy & Anonymization
    pseudonymization_salt: str = os.getenv(
        "PSEUDONYMIZATION_SALT", "academic_research_salt_default"
    )

    # Academic Project Disclaimer
    system_disclaimer: str = (
        "This system provides behavioral estimates and is not a medical diagnostic tool."
    )

    def ensure_directories(self) -> None:
        """Ensure all required project directories exist."""
        for path in [
            self.raw_data_path,
            self.processed_data_path,
            self.sample_data_path,
            self.models_path,
            self.database_path.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Factory function returning application settings."""
    return Settings()


# Default singleton instance for direct import
settings = get_settings()
