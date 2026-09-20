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
    baselines_path: Path = Path(
        os.getenv("BASELINES_PATH", BASE_DIR / "data" / "processed" / "baselines")
    )
    sample_data_path: Path = Path(
        os.getenv("SAMPLE_DATA_PATH", BASE_DIR / "data" / "sample")
    )
    models_path: Path = Path(os.getenv("MODELS_PATH", BASE_DIR / "models"))
    assessments_path: Path = Path(
        os.getenv("ASSESSMENTS_PATH", BASE_DIR / "data" / "processed" / "assessments")
    )

    # Database configuration
    database_path: Path = Path(
        os.getenv("DATABASE_PATH", BASE_DIR / "database" / "mental_state.db")
    )
    database_url: str = os.getenv("DATABASE_URL", "")

    def get_database_url(self) -> str:
        """Resolve the active database URL with environment and Streamlit secrets support.

        Resolution priority:
        1. Explicit DATABASE_URL environment variable
        2. Streamlit secrets (st.secrets['DATABASE_URL'] or st.secrets['postgres']['url'])
        3. Local SQLite fallback (database_path)
        """
        if self.database_url:
            return self.database_url

        # Check Streamlit secrets if running in Streamlit runtime
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                if "DATABASE_URL" in st.secrets:
                    return str(st.secrets["DATABASE_URL"])
                if "database_url" in st.secrets:
                    return str(st.secrets["database_url"])
                if "postgres" in st.secrets and isinstance(st.secrets["postgres"], dict):
                    pg = st.secrets["postgres"]
                    if "url" in pg:
                        return str(pg["url"])
                    user = pg.get("user", "")
                    pwd = pg.get("password", "")
                    host = pg.get("host", "localhost")
                    port = pg.get("port", 5432)
                    dbname = pg.get("dbname", "postgres")
                    return f"postgresql://{user}:{pwd}@{host}:{port}/{dbname}"
        except Exception:
            pass

        return f"sqlite:///{self.database_path.resolve().as_posix()}"

    def get_database_backend(self) -> str:
        """Return the database backend engine type ('sqlite' or 'postgresql')."""
        url = self.get_database_url()
        if url.startswith("postgresql://") or url.startswith("postgres://"):
            return "postgresql"
        return "sqlite"

    # Privacy & Anonymization
    pseudonymization_salt: str = os.getenv(
        "PSEUDONYMIZATION_SALT", "academic_research_salt_default"
    )
    pseudonymization_secret: str = os.getenv(
        "PSEUDONYMIZATION_SECRET",
        os.getenv("PSEUDONYMIZATION_SALT", "academic_research_salt_default"),
    )

    # Cryptographic Storage Key (Fernet 32-byte base64 key)
    encryption_key: str = os.getenv("ENCRYPTION_KEY", "")

    # Differential Privacy Configuration
    dp_enabled: bool = os.getenv("DP_ENABLED", "True").lower() in ("true", "1", "yes")
    dp_epsilon: float = float(os.getenv("DP_EPSILON", "1.0"))

    # Data Retention Schedules (in days)
    raw_event_retention_days: int = int(os.getenv("RAW_EVENT_RETENTION_DAYS", "7"))
    session_retention_days: int = int(os.getenv("SESSION_RETENTION_DAYS", "90"))
    assessment_retention_days: int = int(
        os.getenv("ASSESSMENT_RETENTION_DAYS", "180")
    )

    # Personal Baseline Settings
    min_baseline_sessions: int = int(os.getenv("MIN_BASELINE_SESSIONS", "5"))
    baseline_tolerance: float = float(os.getenv("BASELINE_TOLERANCE", "0.5"))

    # Academic Project Disclaimer
    system_disclaimer: str = (
        "This system provides behavioral estimates and is not a medical diagnostic tool."
    )

    def ensure_directories(self) -> None:
        """Ensure all required project directories exist."""
        for path in [
            self.raw_data_path,
            self.processed_data_path,
            self.baselines_path,
            self.sample_data_path,
            self.models_path,
            self.assessments_path,
            self.database_path.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Factory function returning application settings."""
    return Settings()


# Default singleton instance for direct import
settings = get_settings()
