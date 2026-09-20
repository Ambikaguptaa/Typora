"""Dataset Acquisition and Discovery Status Module.

Inspects the data/raw/ directory to determine whether a real research keystroke
dataset exists, inspects file formats, metadata manifests, and access restrictions.
Enforces ethical acquisition rules: prevents auto-downloading unknown or restricted datasets.

Distinguishes:
- REAL DATASET REQUIRED
- REAL DATASET FOUND BUT INVALID
- REAL DATASET FOUND -- VALIDATION REQUIRED
- REAL DATASET READY FOR TRAINING
"""

from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import zipfile

from src.config.settings import settings

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet", ".json", ".tsv", ".zip"}


class DatasetStatus(str, Enum):
    """Lifecycle status states for dataset acquisition."""

    MISSING = "missing"
    CANDIDATE_FOUND = "candidate_found"
    READY_FOR_VALIDATION = "ready_for_validation"
    VALIDATED = "validated"
    BLOCKED = "blocked"
    INVALID = "invalid"


class DatasetDiscoveryVerdict(str, Enum):
    """High-level staged verdicts for dataset discovery."""

    REQUIRED = "REAL DATASET REQUIRED"
    INVALID = "REAL DATASET FOUND BUT INVALID"
    VALIDATION_REQUIRED = "REAL DATASET FOUND -- VALIDATION REQUIRED"
    READY = "REAL DATASET READY FOR TRAINING"


def get_dataset_status(
    data_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Inspect the raw data directory and report dataset readiness status.

    Args:
        data_dir: Optional custom directory. Defaults to settings.raw_data_path.

    Returns:
        Dict[str, Any]: Structured discovery report detailing files, formats,
        metadata existence, and dataset availability.
    """
    raw_path = Path(data_dir) if data_dir is not None else settings.raw_data_path

    if not raw_path.exists() or not raw_path.is_dir():
        return {
            "status": DatasetStatus.MISSING.value,
            "verdict": DatasetDiscoveryVerdict.REQUIRED.value,
            "data_directory": str(raw_path),
            "candidate_files": [],
            "candidate_paths": [],
            "file_count": 0,
            "file_sizes": {},
            "supported_formats": sorted(list(SUPPORTED_EXTENSIONS)),
            "metadata_files": [],
            "manifest_present": False,
            "labeled_dataset_available": False,
            "labels_detected": [],
            "access_status": "NONE",
            "message": f"Raw data directory '{raw_path}' does not exist. REAL DATASET REQUIRED.",
        }

    # Discover candidate files
    all_files = [f for f in raw_path.iterdir() if f.is_file() and not f.name.startswith(".")]
    candidate_files = [
        f for f in all_files if f.suffix.lower() in SUPPORTED_EXTENSIONS and "manifest" not in f.name.lower()
    ]
    metadata_files = [
        f for f in all_files if "manifest" in f.name.lower() or "metadata" in f.name.lower() or "readme" in f.name.lower()
    ]

    manifest_file = raw_path / "dataset_manifest.json"
    manifest_present = manifest_file.exists() and manifest_file.is_file()

    manifest_info: Dict[str, Any] = {}
    if manifest_present:
        try:
            manifest_info = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    file_sizes = {f.name: f.stat().st_size for f in candidate_files}

    # Determine status and label indicators
    if not candidate_files:
        status = DatasetStatus.MISSING.value
        verdict = DatasetDiscoveryVerdict.REQUIRED.value
        labeled_available = False
        access_status = manifest_info.get("access_status", "NONE")
        message = "No research dataset files found in data/raw/. REAL DATASET REQUIRED."
    else:
        # Check if candidate file is empty or corrupted (e.g. 0 bytes)
        has_corrupt = any(sz == 0 for sz in file_sizes.values())
        if has_corrupt and len(candidate_files) == 1:
            status = DatasetStatus.INVALID.value
            verdict = DatasetDiscoveryVerdict.INVALID.value
            access_status = "INVALID_CORRUPT"
            labeled_available = False
            message = "Candidate dataset file is empty (0 bytes). REAL DATASET FOUND BUT INVALID."
        else:
            # Check if manifest indicates restricted access or pending terms
            access_type = manifest_info.get("access_type", "").upper()
            if "ACCESS REQUIRED" in access_type or manifest_info.get("status") == "blocked":
                status = DatasetStatus.BLOCKED.value
                verdict = DatasetDiscoveryVerdict.REQUIRED.value
                access_status = "ACCESS REQUIRED"
                message = "Candidate dataset requires formal researcher agreement or access credentials."
                labeled_available = False
            else:
                status = DatasetStatus.CANDIDATE_FOUND.value
                verdict = DatasetDiscoveryVerdict.VALIDATION_REQUIRED.value
                access_status = manifest_info.get("access_status", "READY")
                labeled_available = True
                message = f"Found {len(candidate_files)} candidate dataset file(s). REAL DATASET FOUND -- VALIDATION REQUIRED."

    return {
        "status": status,
        "verdict": verdict,
        "data_directory": str(raw_path),
        "candidate_files": [f.name for f in candidate_files],
        "candidate_paths": [str(f) for f in candidate_files],
        "file_count": len(candidate_files),
        "file_sizes": file_sizes,
        "supported_formats": sorted(list(SUPPORTED_EXTENSIONS)),
        "metadata_files": [f.name for f in metadata_files],
        "manifest_present": manifest_present,
        "manifest_info": manifest_info,
        "labeled_dataset_available": labeled_available,
        "labels_detected": manifest_info.get("labels", []),
        "access_status": access_status,
        "message": message,
    }
