"""Application settings and environment configuration.

Loads configuration from .env file or environment variables with sensible defaults.
Designed to be modular, simple, and self-contained.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
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

    def resolve_database_configuration(self) -> Tuple[str, str, str]:
        """Resolve active database URL, backend type, and configuration source with strict precedence.

        Deterministic Precedence:
        1. Streamlit Secrets: st.secrets["DATABASE_URL"] (Uppercase canonical cloud secret)
        2. Streamlit Secrets: st.secrets["database_url"] (Lowercase variant)
        3. Streamlit Secrets: st.secrets["connections"]["postgresql"]["url"]
        4. Streamlit Secrets: st.secrets["postgres"]["url"] or st.secrets["postgres"] table
        5. Streamlit Secrets: st.secrets["postgresql"]["url"]
        6. Environment Variable: os.environ["DATABASE_URL"]
        7. Local SQLite fallback: database/mental_state.db
        """
        # 1-5: Check Streamlit secrets first (authoritative in cloud deployments)
        try:
            import streamlit as st
            secrets = getattr(st, "secrets", None)
            if secrets is not None:
                # 1. Top-level DATABASE_URL
                if "DATABASE_URL" in secrets:
                    val = str(secrets["DATABASE_URL"]).strip().strip("'\"")
                    if val:
                        backend = "postgresql" if val.startswith(("postgresql://", "postgres://")) else "sqlite"
                        return val, backend, "DATABASE_URL FROM STREAMLIT SECRETS"

                # 2. Top-level database_url
                if "database_url" in secrets:
                    val = str(secrets["database_url"]).strip().strip("'\"")
                    if val:
                        backend = "postgresql" if val.startswith(("postgresql://", "postgres://")) else "sqlite"
                        return val, backend, "database_url (lowercase) FROM STREAMLIT SECRETS"

                # 3. connections.postgresql.url
                if "connections" in secrets and hasattr(secrets["connections"], "get"):
                    conns = secrets["connections"]
                    if "postgresql" in conns and hasattr(conns["postgresql"], "get"):
                        pg_conn = conns["postgresql"]
                        if "url" in pg_conn:
                            val = str(pg_conn["url"]).strip().strip("'\"")
                            if val:
                                return val, "postgresql", "connections.postgresql.url FROM STREAMLIT SECRETS"

                # 4. postgres table or dictionary
                if "postgres" in secrets and hasattr(secrets["postgres"], "get"):
                    pg = secrets["postgres"]
                    if "url" in pg:
                        val = str(pg["url"]).strip().strip("'\"")
                        if val:
                            return val, "postgresql", "postgres.url FROM STREAMLIT SECRETS"
                    elif "host" in pg and str(pg["host"]).strip():
                        user = str(pg.get("user", "")).strip()
                        pwd = str(pg.get("password", "")).strip()
                        host = str(pg.get("host", "")).strip()
                        port = int(pg.get("port", 5432))
                        dbname = str(pg.get("dbname", "postgres")).strip()
                        url = f"postgresql://{user}:{pwd}@{host}:{port}/{dbname}"
                        return url, "postgresql", "postgres table FROM STREAMLIT SECRETS"

                # 5. postgresql table or dictionary
                if "postgresql" in secrets and hasattr(secrets["postgresql"], "get"):
                    pg = secrets["postgresql"]
                    if "url" in pg:
                        val = str(pg["url"]).strip().strip("'\"")
                        if val:
                            return val, "postgresql", "postgresql.url FROM STREAMLIT SECRETS"
        except Exception:
            pass

        # 6. Explicit environment variable (from os.environ or loaded .env)
        env_url = (os.environ.get("DATABASE_URL") or self.database_url or "").strip().strip("'\"")
        if env_url:
            backend = "postgresql" if env_url.startswith(("postgresql://", "postgres://")) else "sqlite"
            return env_url, backend, "DATABASE_URL FROM ENVIRONMENT"

        # 7. Local SQLite fallback
        sqlite_url = f"sqlite:///{self.database_path.resolve().as_posix()}"
        return sqlite_url, "sqlite", "LOCAL SQLITE FALLBACK"

    def get_database_url(self) -> str:
        """Resolve the active database URL."""
        url, _, _ = self.resolve_database_configuration()
        return url

    def get_database_backend(self) -> str:
        """Return the database backend engine type ('sqlite' or 'postgresql')."""
        _, backend, _ = self.resolve_database_configuration()
        return backend

    def get_database_source(self) -> str:
        """Return the database configuration source label."""
        _, _, source = self.resolve_database_configuration()
        return source

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
