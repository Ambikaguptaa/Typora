"""Storage Policy Module.

Enforces cryptographic, pseudonymization, and retention requirements for all
stored data artifacts and reports in accordance with academic research governance.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.config.settings import settings
from src.privacy.data_classification import DataSensitivityLevel


@dataclass(frozen=True)
class StoragePolicy:
    """Specification of storage, encryption, and retention rules for an artifact type."""

    artifact_type: str
    sensitivity_level: DataSensitivityLevel
    encryption_required: bool
    pseudonymization_required: bool
    retention_days: int
    allow_raw_text: bool = False  # Strict Zero-Text invariant: Always False
    description: str = ""


def get_default_storage_policies() -> Dict[str, StoragePolicy]:
    """Construct registry of storage policies across all artifact types."""
    return {
        "raw_keystrokes": StoragePolicy(
            artifact_type="raw_keystrokes",
            sensitivity_level=DataSensitivityLevel.HIGHLY_SENSITIVE,
            encryption_required=True,
            pseudonymization_required=True,
            retention_days=settings.raw_event_retention_days,
            allow_raw_text=False,
            description="Raw keystroke timing events metadata (zero character text).",
        ),
        "timing_features": StoragePolicy(
            artifact_type="timing_features",
            sensitivity_level=DataSensitivityLevel.SENSITIVE,
            encryption_required=False,
            pseudonymization_required=True,
            retention_days=settings.session_retention_days,
            allow_raw_text=False,
            description="Aggregated keystroke timing features (dwell, flight, pauses).",
        ),
        "personal_baselines": StoragePolicy(
            artifact_type="personal_baselines",
            sensitivity_level=DataSensitivityLevel.SENSITIVE,
            encryption_required=True,
            pseudonymization_required=True,
            retention_days=settings.session_retention_days,
            allow_raw_text=False,
            description="Participant baseline behavioral profiles and deviation indices.",
        ),
        "model_checkpoints": StoragePolicy(
            artifact_type="model_checkpoints",
            sensitivity_level=DataSensitivityLevel.INTERNAL,
            encryption_required=False,
            pseudonymization_required=False,
            retention_days=365,
            allow_raw_text=False,
            description="Neural network weights, scalers, and architecture configs.",
        ),
        "assessment_reports": StoragePolicy(
            artifact_type="assessment_reports",
            sensitivity_level=DataSensitivityLevel.SENSITIVE,
            encryption_required=True,
            pseudonymization_required=True,
            retention_days=settings.assessment_retention_days,
            allow_raw_text=False,
            description="Behavioral assessment estimates, reliability scores, and interpretations.",
        ),
        "audit_logs": StoragePolicy(
            artifact_type="audit_logs",
            sensitivity_level=DataSensitivityLevel.INTERNAL,
            encryption_required=False,
            pseudonymization_required=False,
            retention_days=365,
            allow_raw_text=False,
            description="Security, operational, and data governance audit logs.",
        ),
    }


def get_storage_policy(artifact_type: str) -> StoragePolicy:
    """Retrieve storage policy for a specified artifact type.

    Args:
        artifact_type: Name/category of the artifact.

    Returns:
        StoragePolicy: Applicable storage policy.
    """
    policies = get_default_storage_policies()
    cleaned = artifact_type.strip().lower()
    if cleaned in policies:
        return policies[cleaned]

    # Default conservative policy for unrecognized artifacts
    return StoragePolicy(
        artifact_type=cleaned,
        sensitivity_level=DataSensitivityLevel.SENSITIVE,
        encryption_required=True,
        pseudonymization_required=True,
        retention_days=settings.session_retention_days,
        allow_raw_text=False,
        description=f"Generic fallback storage policy for {cleaned}.",
    )


def validate_artifact_storage_compliance(
    artifact_type: str,
    has_raw_text: bool,
    is_pseudonymized: bool,
    is_encrypted: bool,
) -> Tuple[bool, List[str]]:
    """Verify that an artifact's storage status complies with its defined policy.

    Args:
        artifact_type: Type of the artifact.
        has_raw_text: Whether raw keystroke character text is present.
        is_pseudonymized: Whether user identifiers have been pseudonymized.
        is_encrypted: Whether data is encrypted at rest.

    Returns:
        Tuple[bool, List[str]]:
            - bool: True if compliant, False otherwise.
            - List[str]: List of violation descriptions if any.
    """
    policy = get_storage_policy(artifact_type)
    violations: List[str] = []

    if has_raw_text and not policy.allow_raw_text:
        violations.append(
            f"Zero-Text Policy violation: Artifact '{artifact_type}' contains forbidden raw character text."
        )

    if policy.pseudonymization_required and not is_pseudonymized:
        violations.append(
            f"Pseudonymization violation: Artifact '{artifact_type}' requires pseudonymized user identifiers."
        )

    if policy.encryption_required and not is_encrypted:
        violations.append(
            f"Encryption violation: Artifact '{artifact_type}' requires authenticated encryption at rest."
        )

    return (len(violations) == 0, violations)
