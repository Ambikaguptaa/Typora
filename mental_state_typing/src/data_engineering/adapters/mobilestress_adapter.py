"""MobileStress Research Dataset Adapter Module.

Adapts smartphone typing dynamics from the MobileStress experimental protocol into the
system's canonical event schema.
Subclasses BaseDatasetAdapter.
Enforces strict Zero-Raw-Text suppression and preserves dataset condition semantics.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data_engineering.adapters.base_adapter import BaseDatasetAdapter
from src.data_engineering.dataset_adapter import load_dataset
from src.privacy.privacy_utils import strip_character_data
from src.privacy.pseudonymization import is_valid_pseudonym, pseudonymize_user_id

# Documented MobileStress experimental condition mappings
DEFAULT_MOBILESTRESS_CONDITIONS = {
    "E": "neutral",   # Baseline typing rhythm under standard task condition
    "H": "stressed",  # Typing rhythm under induced cognitive/temporal stress
    "0": "neutral",
    "1": "stressed",
    "neutral": "neutral",
    "stressed": "stressed",
    "stress": "stressed",
    "baseline": "neutral",
}


class MobileStressAdapter(BaseDatasetAdapter):
    """Adapter translating MobileStress schema into standard canonical keystroke format."""

    def __init__(
        self,
        condition_mapping: Optional[Dict[str, str]] = None,
        pseudonymize_users: bool = True,
        secret: Optional[str] = None,
    ):
        """Initialize adapter.

        Args:
            condition_mapping: Optional dictionary mapping dataset condition codes to standard labels.
            pseudonymize_users: Whether to apply HMAC pseudonymization to participant identifiers.
            secret: Optional HMAC secret.
        """
        self.condition_mapping = condition_mapping or DEFAULT_MOBILESTRESS_CONDITIONS
        self.pseudonymize_users = pseudonymize_users
        self.secret = secret

    def detect(self, source: Union[str, Path, pd.DataFrame]) -> bool:
        """Heuristically detect if source matches MobileStress format.

        Args:
            source: File path or DataFrame.

        Returns:
            bool: True if source exhibits MobileStress column or naming markers.
        """
        if isinstance(source, (str, Path)):
            p = Path(source)
            if "mobilestress" in p.name.lower():
                return True
            try:
                df = load_dataset(p)
            except Exception:
                return False
        elif isinstance(source, pd.DataFrame):
            df = source
        else:
            return False

        cols_lower = {str(c).strip().lower() for c in df.columns}
        has_user = any(k in cols_lower for k in ["participant", "participant_id", "subject", "sub_id", "uid", "user_id"])
        has_time = any(k in cols_lower for k in ["time", "timestamp", "down_time", "press_time"])
        has_cond = any(k in cols_lower for k in ["condition", "stress", "task", "state"])

        return has_user and has_time and has_cond

    def load(self, source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load raw MobileStress observations.

        Args:
            source: Path or DataFrame.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        if isinstance(source, pd.DataFrame):
            return source.copy()
        return load_dataset(Path(source))

    def normalize(self, source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Conform raw observations into the Canonical Event Schema.

        Args:
            source: Raw file path or DataFrame.

        Returns:
            pd.DataFrame: Canonical event DataFrame.
        """
        return self.adapt(source)

    def adapt(
        self,
        source: Union[str, Path, pd.DataFrame],
    ) -> pd.DataFrame:
        """Load and transform raw MobileStress observations into canonical schema.

        Args:
            source: Path to raw dataset file (.csv, .xlsx, etc.) or raw DataFrame.

        Returns:
            pd.DataFrame: Normalized DataFrame conforming to internal canonical schema:
                [participant_id, session_id, timestamp, event_type, key_identifier,
                 press_time, release_time, condition] (+ optional pressure, x, y).
        """
        df = self.load(source)

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "participant_id",
                    "session_id",
                    "timestamp",
                    "event_type",
                    "key_identifier",
                    "press_time",
                    "release_time",
                    "condition",
                ]
            )

        # 1. Column detection & renaming
        cols_lower = {str(c): str(c).strip().lower() for c in df.columns}
        inv_cols = {v: k for k, v in cols_lower.items()}

        # Participant ID
        user_col = None
        for cand in ["participant_id", "participant", "user_id", "subject", "sub_id", "uid"]:
            if cand in inv_cols:
                user_col = inv_cols[cand]
                break

        # Session ID
        sess_col = None
        for cand in ["session", "session_id", "sess", "trial", "trial_id"]:
            if cand in inv_cols:
                sess_col = inv_cols[cand]
                break

        # Timestamp
        time_col = None
        for cand in ["time", "timestamp", "down_time", "press_time", "datetime"]:
            if cand in inv_cols:
                time_col = inv_cols[cand]
                break

        # Condition / Label
        cond_col = None
        for cand in ["condition", "state", "stress", "label", "task"]:
            if cand in inv_cols:
                cond_col = inv_cols[cand]
                break

        # Event type (down/up)
        event_col = None
        for cand in ["event_type", "type", "action", "event"]:
            if cand in inv_cols:
                event_col = inv_cols[cand]
                break

        # Key (to be masked)
        key_col = None
        for cand in ["key", "keycode", "key_id", "char"]:
            if cand in inv_cols:
                key_col = inv_cols[cand]
                break

        if not user_col or not time_col:
            raise ValueError(
                f"Required MobileStress columns missing: user_col={user_col}, time_col={time_col}"
            )

        # 2. Sort by participant, session, time
        sort_keys = [c for c in [user_col, sess_col, time_col] if c is not None]
        df = df.sort_values(by=sort_keys).reset_index(drop=True)

        # 3. Format participant IDs (pseudonymize if requested)
        participant_series = df[user_col].astype(str)
        if self.pseudonymize_users:
            participant_series = participant_series.apply(
                lambda u: u if is_valid_pseudonym(u) else pseudonymize_user_id(u, secret=self.secret)
            )

        # 4. Format session IDs
        if sess_col:
            session_series = df[sess_col].astype(str)
        else:
            session_series = pd.Series(["sess_0"] * len(df))

        # 5. Format timestamps
        timestamp_series = pd.to_numeric(df[time_col], errors="coerce").fillna(0.0)

        # 6. Format condition / label
        if cond_col:
            condition_series = df[cond_col].astype(str).map(
                lambda c: self.condition_mapping.get(c, self.condition_mapping.get(c.upper(), c))
            )
        else:
            condition_series = pd.Series(["unknown"] * len(df))

        # 7. Mask key identities into zero-text key identifiers (e.g. k_hash)
        if key_col:
            # We never store raw character strings; hash or mask to numeric/abstract ID
            key_identifier_series = df[key_col].astype(str).apply(
                lambda k: f"k_{hashlib.md5(k.encode('utf-8')).hexdigest()[:6]}"
            )
        else:
            key_identifier_series = pd.Series(["k_unk"] * len(df))

        # 8. Event type and timing pairing
        event_series = (
            df[event_col].astype(str).str.lower()
            if event_col
            else pd.Series(["down"] * len(df))
        )

        # Build normalized base DataFrame
        norm_df = pd.DataFrame(
            {
                "participant_id": participant_series,
                "session_id": session_series,
                "timestamp": timestamp_series,
                "event_type": event_series,
                "key_identifier": key_identifier_series,
                "condition": condition_series,
            }
        )

        # Optional spatial/pressure telemetry
        for opt_col in ["pressure", "x", "y"]:
            if opt_col in inv_cols:
                norm_df[opt_col] = pd.to_numeric(df[inv_cols[opt_col]], errors="coerce")

        # 9. Derive key-down / key-up timing where possible
        has_down_up = any("up" in ev for ev in event_series.unique()) and any(
            "down" in ev for ev in event_series.unique()
        )

        if has_down_up:
            press_times, release_times = self._pair_down_up_events(norm_df)
            norm_df["press_time"] = press_times
            norm_df["release_time"] = release_times
            # Filter to down events with paired release times or retain all
            norm_df = norm_df[norm_df["event_type"].str.contains("down")].copy()
        else:
            # Only single timestamp per event (e.g. only down events)
            norm_df["press_time"] = norm_df["timestamp"]
            norm_df["release_time"] = np.nan  # Do not fabricate release times

        # 10. Strictly strip any accidental raw text columns
        norm_df = strip_character_data(norm_df)

        return norm_df.reset_index(drop=True)

    def _pair_down_up_events(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series]:
        """Pair sequential DOWN and UP events for each key within participant sessions."""
        press_times = []
        release_times = []

        pending_downs: Dict[Tuple[str, str, str], Tuple[int, float]] = {}
        down_to_up_map: Dict[int, float] = {}

        for idx, row in df.iterrows():
            p_id = row["participant_id"]
            s_id = row["session_id"]
            k_id = row["key_identifier"]
            t = row["timestamp"]
            ev = str(row["event_type"]).lower()

            key_tuple = (p_id, s_id, k_id)

            if "down" in ev:
                pending_downs[key_tuple] = (idx, float(t))
            elif "up" in ev and key_tuple in pending_downs:
                down_idx, down_time = pending_downs.pop(key_tuple)
                down_to_up_map[down_idx] = float(t)

        for idx, row in df.iterrows():
            t = float(row["timestamp"])
            press_times.append(t)
            release_times.append(down_to_up_map.get(idx, np.nan))

        return pd.Series(press_times, index=df.index), pd.Series(release_times, index=df.index)

    def get_metadata(self) -> Dict[str, Any]:
        """Return MobileStress adapter metadata."""
        return {
            "adapter_name": "MobileStressAdapter",
            "dataset_name": "MobileStress Keystroke Dataset",
            "access_status": "ACCESS_REQUIRED",
            "source": "Academic Keystroke Research Repository",
            "canonical_schema_compliant": True,
            "pseudonymization_enabled": self.pseudonymize_users,
            "zero_raw_text_enforced": True,
            "citation": (
                "MobileStress: Smartphone Keystroke Dynamics Under Induced Cognitive and Temporal Stress. "
                "Research Study."
            ),
        }

    def get_label_mapping(self) -> Dict[str, str]:
        """Return MobileStress condition mappings."""
        return dict(self.condition_mapping)

    def get_supported_features(self) -> List[str]:
        """Return supported keystroke timing features derivable from MobileStress."""
        return [
            "dwell_time",
            "flight_time",
            "pause_duration",
            "typing_speed",
            "backspace",
        ]


def adapt_mobilestress_dataset(
    source: Union[str, Path, pd.DataFrame],
    secret: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience functional wrapper to adapt a MobileStress dataset."""
    adapter = MobileStressAdapter(secret=secret)
    return adapter.adapt(source)
