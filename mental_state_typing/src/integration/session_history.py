"""Session History and Stored Assessment Manager Module.

Discovers, loads, audits, and presents historical typing session assessments
stored in data/processed/assessments/.
Enforces strict privacy auditing so no raw text or sensitive payloads are exposed.
Handles corrupted or malformed files gracefully without application crashes.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.config.settings import settings
from src.integration.result_schema import FORBIDDEN_PAYLOAD_FIELDS
from src.live_typing.privacy_filter import PrivacyViolationError, audit_payload_for_sensitive_keys

logger = logging.getLogger(__name__)


def get_assessments_directory(assessments_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve and ensure the assessments directory exists."""
    if assessments_dir is not None:
        p = Path(assessments_dir)
    else:
        p = settings.processed_data_path / "assessments"
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_stored_sessions(
    assessments_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Scan assessments directory and return privacy-safe metadata summaries.

    Args:
        assessments_dir: Optional custom assessments path. Defaults to settings.

    Returns:
        List[Dict[str, Any]]: Chronologically sorted list of session summaries (newest first).
    """
    target_dir = get_assessments_directory(assessments_dir)
    sessions: List[Dict[str, Any]] = []

    if not target_dir.exists():
        return sessions

    json_files = sorted(target_dir.glob("assessment_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

    for file_path in json_files:
        session_id = file_path.stem.replace("assessment_", "")
        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        try:
            content = file_path.read_text(encoding="utf-8")
            data = json.loads(content)

            # Audit against forbidden raw text keys
            violations = audit_payload_for_sensitive_keys(data)
            if violations:
                raise PrivacyViolationError(f"Forbidden raw text keys detected: {violations}")

            # Extract privacy-safe summary attributes
            sid = data.get("session_id", session_id)
            stype = data.get("session_type", "ANALYSIS")
            ts = data.get("timestamp", file_mtime)

            dq = data.get("data_quality", {})
            quality_verdict = dq.get("verdict", "UNKNOWN")
            is_valid = dq.get("is_valid", False)
            metrics = dq.get("metrics", {})
            duration = metrics.get("active_duration_s", 0.0)
            events = metrics.get("paired_events_count", 0)

            base_res = data.get("baseline_result", {})
            base_status = base_res.get("status", "NOT_READY")
            tdi_val = base_res.get("typing_deviation_index")

            model_res = data.get("model_result", {})
            model_status = model_res.get("status", "MODEL_NOT_READY")
            pred_class = model_res.get("predicted_class")

            sessions.append(
                {
                    "session_id": sid,
                    "file_name": file_path.name,
                    "date": ts[:19].replace("T", " ") if "T" in str(ts) else str(ts),
                    "session_type": stype,
                    "duration_s": round(float(duration), 1),
                    "event_count": int(events),
                    "quality": "VALID" if is_valid else "INVALID",
                    "quality_verdict": quality_verdict,
                    "baseline_status": base_status,
                    "tdi": round(float(tdi_val), 1) if tdi_val is not None else None,
                    "model_status": model_status,
                    "predicted_class": pred_class,
                    "is_corrupted": False,
                }
            )

        except json.JSONDecodeError as jde:
            logger.warning(f"Corrupted assessment JSON in {file_path.name}: {jde}")
            sessions.append(
                {
                    "session_id": session_id,
                    "file_name": file_path.name,
                    "date": file_mtime,
                    "session_type": "UNKNOWN",
                    "duration_s": 0.0,
                    "event_count": 0,
                    "quality": "CORRUPTED",
                    "quality_verdict": "CORRUPTED_JSON",
                    "baseline_status": "N/A",
                    "tdi": None,
                    "model_status": "N/A",
                    "predicted_class": None,
                    "is_corrupted": True,
                    "error_message": f"Malformed JSON: {jde}",
                }
            )
        except PrivacyViolationError as pve:
            logger.error(f"Privacy violation in assessment file {file_path.name}: {pve}")
            sessions.append(
                {
                    "session_id": session_id,
                    "file_name": file_path.name,
                    "date": file_mtime,
                    "session_type": "BLOCKED",
                    "duration_s": 0.0,
                    "event_count": 0,
                    "quality": "BLOCKED",
                    "quality_verdict": "PRIVACY_VIOLATION",
                    "baseline_status": "BLOCKED",
                    "tdi": None,
                    "model_status": "BLOCKED",
                    "predicted_class": None,
                    "is_corrupted": True,
                    "error_message": f"File contains forbidden raw text fields: {pve}",
                }
            )
        except Exception as e:
            logger.warning(f"Unexpected error loading assessment {file_path.name}: {e}")
            sessions.append(
                {
                    "session_id": session_id,
                    "file_name": file_path.name,
                    "date": file_mtime,
                    "session_type": "ERROR",
                    "duration_s": 0.0,
                    "event_count": 0,
                    "quality": "ERROR",
                    "quality_verdict": "LOAD_ERROR",
                    "baseline_status": "ERROR",
                    "tdi": None,
                    "model_status": "ERROR",
                    "predicted_class": None,
                    "is_corrupted": True,
                    "error_message": str(e),
                }
            )

    # In live deployment (when assessments_dir is None), check persistent database for additional records
    if assessments_dir is None:
        try:
            from database.database import get_assessment_records

            existing_sids = {s["session_id"] for s in sessions}
            db_records = get_assessment_records()
            for rec in db_records:
                sid = rec.get("session_id")
                if sid and sid not in existing_sids:
                    sessions.append(
                        {
                            "session_id": sid,
                            "file_name": f"assessment_{sid}.json",
                            "date": str(rec.get("created_at", ""))[:19].replace("T", " "),
                            "session_type": rec.get("session_type", "ANALYSIS"),
                            "duration_s": round(float(rec.get("duration_s", 0.0)), 1),
                            "event_count": int(rec.get("event_count", 0)),
                            "quality": "VALID",
                            "quality_verdict": "PASS",
                            "baseline_status": "READY" if rec.get("tdi_score") is not None else "NOT_READY",
                            "tdi": round(float(rec["tdi_score"]), 1) if rec.get("tdi_score") is not None else None,
                            "model_status": rec.get("model_status", "MODEL_NOT_READY"),
                            "predicted_class": rec.get("model_predicted_class"),
                            "is_corrupted": False,
                        }
                    )
                    existing_sids.add(sid)
        except Exception as e:
            logger.warning(f"Could not load fallback assessments from database: {e}")

    return sessions


def load_session_detail(
    session_id: str,
    assessments_dir: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Load and privacy-audit a single session assessment file.

    Args:
        session_id: The session UUID or filename stem.
        assessments_dir: Optional custom assessments path.

    Returns:
        Optional[Dict[str, Any]]: The parsed assessment dictionary, or None if missing/corrupted.

    Raises:
        PrivacyViolationError: If sensitive keys are detected.
    """
    target_dir = get_assessments_directory(assessments_dir)

    # Resolve filename
    clean_id = session_id.replace("assessment_", "").replace(".json", "")
    target_file = target_dir / f"assessment_{clean_id}.json"

    if not target_file.exists():
        # Try direct match
        target_file = target_dir / f"{session_id}.json"
        if not target_file.exists():
            # In live mode, fallback to persistent database
            if assessments_dir is None:
                try:
                    from database.database import get_assessment_by_id

                    db_detail = get_assessment_by_id(clean_id)
                    if db_detail:
                        return db_detail
                except Exception as e:
                    logger.warning(f"Failed loading assessment from database: {e}")

            logger.warning(f"Session assessment file not found: {target_file}")
            return None

    try:
        content = target_file.read_text(encoding="utf-8")
        data = json.loads(content)

        # Audit against sensitive text keys
        violations = audit_payload_for_sensitive_keys(data)
        if violations:
            raise PrivacyViolationError(f"Forbidden raw text keys in assessment: {violations}")

        return data
    except PrivacyViolationError:
        raise
    except Exception as e:
        logger.warning(f"Failed loading session detail for {session_id}: {e}")
        return None


def delete_session_record(
    session_id: str,
    assessments_dir: Optional[Union[str, Path]] = None,
) -> bool:
    """Delete a session assessment record (data minimization / GDPR compliance).

    Args:
        session_id: The session identifier.
        assessments_dir: Optional custom assessments directory.

    Returns:
        bool: True if file or DB record was deleted, False otherwise.
    """
    target_dir = get_assessments_directory(assessments_dir)
    clean_id = session_id.replace("assessment_", "").replace(".json", "")
    target_file = target_dir / f"assessment_{clean_id}.json"

    file_deleted = False
    if target_file.exists():
        try:
            target_file.unlink()
            file_deleted = True
        except Exception as e:
            logger.warning(f"Failed deleting session file {target_file}: {e}")

    # In live mode, also purge from database
    if assessments_dir is None:
        try:
            from database.database import delete_assessment_record

            db_deleted = delete_assessment_record(clean_id)
            return file_deleted or db_deleted
        except Exception as e:
            logger.warning(f"Failed deleting assessment from database: {e}")

    return file_deleted
