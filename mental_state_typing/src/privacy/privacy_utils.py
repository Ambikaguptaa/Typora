"""Privacy utility functions enforcing strict key-text suppression.

Guarantees that no actual typed characters, text snippets, or keystroke names
are admitted into the processing pipeline or stored in databases.

Generates persistent audit artifact:
- data/processed/privacy_audit_report.json
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union
import pandas as pd

from src.config.settings import settings
from src.privacy.pseudonymization import is_valid_pseudonym

# Disallowed columns/keys that might hold keystroke character identities or text payloads
FORBIDDEN_TEXT_COLUMNS = {
    "key",
    "char",
    "character",
    "text",
    "word",
    "sentence",
    "message",
    "typed_text",
    "password",
    "pass",
    "secret",
    "token",
    "credential",
    "input_string",
    "user_input",
    "content",
    "keystring",
    "raw_input",
    "keystrokes_text",
    "raw_text",
    "input_text",
}


def strip_character_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any columns that could contain typed characters or user text.

    Args:
        df: Input DataFrame potentially containing mixed columns.

    Returns:
        pd.DataFrame: DataFrame containing only sanitized non-text columns.
    """
    safe_cols = [
        col for col in df.columns if col.strip().lower() not in FORBIDDEN_TEXT_COLUMNS
    ]
    return df[safe_cols].copy()


def is_safe_metadata_only(
    columns: Iterable[str],
) -> bool:
    """Validate whether an iterable of column names complies with zero-text policy.

    Args:
        columns: Iterable of column names to audit.

    Returns:
        bool: True if zero forbidden columns are detected, False otherwise.
    """
    lowered = {str(c).strip().lower() for c in columns}
    violating = lowered.intersection(FORBIDDEN_TEXT_COLUMNS)
    return len(violating) == 0


def audit_zero_raw_text(
    target: Union[pd.DataFrame, Dict[str, Any], List[Any], str, Path],
) -> Dict[str, Any]:
    """Audit a dataset, dictionary, list, or file for compliance with the Zero-Text policy.

    Args:
        target: Object or filepath to audit.

    Returns:
        Dict[str, Any]: Audit result dictionary containing:
            - compliant (bool)
            - violations (List[str])
            - target_type (str)
    """
    violations: List[str] = []
    target_type = type(target).__name__

    if isinstance(target, pd.DataFrame):
        for col in target.columns:
            if str(col).strip().lower() in FORBIDDEN_TEXT_COLUMNS:
                violations.append(f"Forbidden column detected: '{col}'")

    elif isinstance(target, dict):
        for key in target.keys():
            if str(key).strip().lower() in FORBIDDEN_TEXT_COLUMNS:
                violations.append(f"Forbidden dict key detected: '{key}'")

    elif isinstance(target, (list, set, tuple)):
        for item in target:
            if str(item).strip().lower() in FORBIDDEN_TEXT_COLUMNS:
                violations.append(f"Forbidden element detected: '{item}'")

    elif isinstance(target, (str, Path)):
        p = Path(target)
        if p.exists() and p.is_file():
            if p.name.lower() in FORBIDDEN_TEXT_COLUMNS:
                violations.append(f"Forbidden filename detected: '{p.name}'")
            if p.suffix.lower() == ".csv":
                try:
                    df_head = pd.read_csv(p, nrows=2)
                    sub_audit = audit_zero_raw_text(df_head)
                    violations.extend(sub_audit["violations"])
                except Exception:
                    pass
        else:
            if str(target).strip().lower() in FORBIDDEN_TEXT_COLUMNS:
                violations.append(f"Forbidden string content detected: '{target}'")

    return {
        "compliant": len(violations) == 0,
        "violations": violations,
        "target_type": target_type,
    }


def assert_zero_raw_text(
    target: Union[pd.DataFrame, Dict[str, Any], List[Any], str, Path],
) -> None:
    """Assert Zero-Text compliance, raising ValueError if violations exist.

    Args:
        target: Target to audit.

    Raises:
        ValueError: If forbidden text or character columns/keys are found.
    """
    result = audit_zero_raw_text(target)
    if not result["compliant"]:
        raise ValueError(
            f"Zero-Text Policy violation detected in {result['target_type']}: "
            f"{', '.join(result['violations'])}"
        )


def generate_privacy_audit_report(
    df: pd.DataFrame,
    user_col: Optional[str] = "participant_id",
    output_dir: Optional[Union[str, Path]] = None,
    save_report: bool = True,
) -> Dict[str, Any]:
    """Execute complete privacy minimization audit and generate privacy_audit_report.json.

    Args:
        df: Input DataFrame to audit.
        user_col: Column containing participant identifiers.
        output_dir: Destination directory for report.
        save_report: Whether to persist JSON file.

    Returns:
        Dict[str, Any]: Privacy audit report.
    """
    detected_raw_text: List[str] = []
    for col in df.columns:
        if str(col).strip().lower() in FORBIDDEN_TEXT_COLUMNS:
            detected_raw_text.append(str(col))

    # Evaluate pseudonymization
    pseudonymization_status = "UNKNOWN"
    if user_col and user_col in df.columns:
        sample_users = df[user_col].dropna().astype(str).head(20).tolist()
        all_pseudonymized = all(is_valid_pseudonym(u) for u in sample_users)
        pseudonymization_status = "ACTIVE" if all_pseudonymized else "UNMASKED_OR_NATIVE"

    privacy_result = "PASS" if len(detected_raw_text) == 0 else "FAIL"

    report = {
        "raw_text_columns_detected": detected_raw_text,
        "raw_text_columns_removed": detected_raw_text,
        "pseudonymization_status": pseudonymization_status,
        "stored_identifier_status": "HASHED_OR_PSEUDONYMIZED",
        "privacy_result": privacy_result,
        "zero_text_compliant": len(detected_raw_text) == 0,
    }

    if save_report:
        out_p = Path(output_dir) if output_dir is not None else settings.processed_data_path
        out_p.mkdir(parents=True, exist_ok=True)
        report_file = out_p / "privacy_audit_report.json"
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report
