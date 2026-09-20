"""CMU Keystroke Dynamics Dataset Adapter Specification / Stub.

Documents the adapter specification for the Carnegie Mellon University (CMU)
Keystroke Dynamics benchmark dataset (Killourhy & Maxion).

IMPORTANT ENGINEERING RULE:
The CMU dataset is not currently available in data/raw/.
In accordance with scientific integrity guidelines:
- No columns are fabricated.
- No synthetic labels are invented.
- This adapter acts as a documented specification until an approved real CMU dataset file is obtained.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from src.data_engineering.adapters.base_adapter import BaseDatasetAdapter


class CMUStressAdapter(BaseDatasetAdapter):
    """Adapter specification for CMU Keystroke Dynamics benchmark.

    Status: SPECIFICATION_ONLY (Access required / Dataset not installed).
    """

    def __init__(self, condition_mapping: Optional[Dict[str, str]] = None):
        """Initialize CMU adapter stub."""
        self.condition_mapping = condition_mapping or {}

    def detect(self, source: Union[str, Path, pd.DataFrame]) -> bool:
        """Check if source matches CMU Keystroke Benchmark characteristics."""
        if isinstance(source, (str, Path)):
            p = Path(source)
            if "cmu" in p.name.lower() or "dsl-strongpassword" in p.name.lower():
                return True
        return False

    def load(self, source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Attempt to load raw CMU observations.

        Raises:
            FileNotFoundError: Explaining that CMU dataset must be acquired and placed in data/raw/.
        """
        if isinstance(source, pd.DataFrame):
            return source.copy()

        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(
                f"CMU dataset file not found at: {p}. "
                "The CMU keystroke dynamics dataset requires formal download/acquisition "
                "from the official Carnegie Mellon University repository before it can be loaded."
            )
        return pd.read_csv(p)

    def normalize(self, source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Transform raw CMU observations into Canonical Event Schema.

        Raises:
            NotImplementedError: Awaiting verified local dataset schema to prevent column fabrication.
        """
        raise NotImplementedError(
            "CMU dataset normalization is pending local schema verification. "
            "To prevent scientific invalidity, column transformations will only be enabled "
            "once the official research dataset is deposited in data/raw/ and verified."
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Return CMU adapter metadata and scientific citation."""
        return {
            "adapter_name": "CMUStressAdapter",
            "dataset_name": "CMU Keystroke Dynamics Benchmark Dataset",
            "access_status": "ACCESS_REQUIRED",
            "source": "Carnegie Mellon University (Killourhy & Maxion)",
            "canonical_schema_compliant": False,
            "status": "SPECIFICATION_STUB",
            "citation": (
                "Killourhy, K. S., & Maxion, R. A. (2009). "
                "Comparing anomaly-detection algorithms for keystroke dynamics. "
                "IEEE/IFIP International Conference on Dependable Systems & Networks (DSN)."
            ),
        }

    def get_label_mapping(self) -> Dict[str, str]:
        """Return label mappings (empty until verified)."""
        return dict(self.condition_mapping)

    def get_supported_features(self) -> List[str]:
        """Return timing features supported by CMU benchmark."""
        return [
            "dwell_time",
            "flight_time",
            "latency",
        ]
