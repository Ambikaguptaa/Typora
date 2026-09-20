"""Abstract Base Dataset Adapter Interface.

Defines the pluggable interface for keystroke dynamics dataset adapters.
Every concrete adapter (e.g. MobileStressAdapter, CMUStressAdapter) must implement
this contract to convert heterogeneous external research formats into the system's
internal Canonical Event Schema.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from src.data_engineering.canonical_schema import (
    CANONICAL_REQUIRED_COLUMNS,
    validate_canonical_dataframe,
)


class BaseDatasetAdapter(ABC):
    """Abstract base class defining the required contract for dataset adapters."""

    @abstractmethod
    def detect(self, source: Union[str, Path, pd.DataFrame]) -> bool:
        """Inspect source file or DataFrame and return True if compatible with this adapter.

        Args:
            source: File path or pandas DataFrame to inspect.

        Returns:
            bool: True if source matches this adapter's schema heuristics.
        """
        pass

    @abstractmethod
    def load(self, source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load raw dataset observations into an unnormalized pandas DataFrame.

        Args:
            source: File path or existing DataFrame.

        Returns:
            pd.DataFrame: Loaded raw DataFrame.
        """
        pass

    @abstractmethod
    def normalize(self, source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Transform raw observations into the system's Canonical Event Schema.

        Enforces:
        - Mapping to canonical column names:
          [participant_id, session_id, timestamp, event_type, key_identifier,
           press_time, release_time, condition] (+ optional pressure, x, y).
        - Zero-Raw-Text suppression: raw characters hashed or masked to abstract tokens.
        - Pseudonymization of participant identifiers.
        - Experimental condition mapping to canonical behavioral labels.

        Args:
            source: Raw file path or raw DataFrame.

        Returns:
            pd.DataFrame: Clean DataFrame conforming to canonical event schema.
        """
        pass

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate an adapted DataFrame against canonical schema requirements.

        Default implementation leverages `validate_canonical_dataframe`.

        Args:
            df: Adapted DataFrame to validate.

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_error_messages).
        """
        report = validate_canonical_dataframe(df)
        return report["is_valid"], report["errors"]

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return adapter metadata, including dataset name, citation, and license.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        pass

    @abstractmethod
    def get_label_mapping(self) -> Dict[str, str]:
        """Return the dictionary mapping raw dataset labels to canonical conditions.

        Returns:
            Dict[str, str]: Label mapping (e.g. {"E": "neutral", "H": "stressed"}).
        """
        pass

    @abstractmethod
    def get_supported_features(self) -> List[str]:
        """Return list of timing features legitimately derivable from this dataset format.

        Returns:
            List[str]: Supported timing feature names.
        """
        pass
