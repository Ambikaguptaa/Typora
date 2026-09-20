"""Live typing feature buffer and temporal sequence windowing.

Maintains a rolling buffer of normalized TypingEvents and computes:
1. Real-time session telemetry summary (mean dwell, flight, pauses, speed, backspaces).
2. Canonical event feature DataFrame matching offline data engineering expectations.
3. 3D temporal sequence arrays (num_windows, sequence_length, num_features) ready for LSTM inference.

ZERO-TEXT INVARIANT:
All features are strictly numeric and behavioral. No character information is preserved or extracted.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.live_typing.event_types import TypingEvent


CANONICAL_SEQUENCE_FEATURES = [
    "dwell_time",
    "flight_time",
    "pause_duration",
    "typing_speed",
    "backspace",
    "error_flag",
]


class LiveFeatureBuffer:
    """Rolling buffer of normalized TypingEvents providing feature extraction and temporal windowing."""

    def __init__(
        self,
        max_buffer_size: int = 1000,
        pause_threshold_ms: float = 500.0,
        sequence_length: int = 30,
        sequence_stride: int = 10,
    ):
        self.max_buffer_size = max_buffer_size
        self.pause_threshold_ms = pause_threshold_ms
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self._events: List[TypingEvent] = []

    def clear(self) -> None:
        """Clear all buffered events."""
        self._events.clear()

    def add_event(self, event: TypingEvent) -> None:
        """Append a normalized typing event, evicting oldest if capacity exceeded."""
        self._events.append(event)
        if len(self._events) > self.max_buffer_size:
            self._events.pop(0)

    def add_events(self, events: List[TypingEvent]) -> None:
        """Append multiple normalized typing events."""
        for ev in events:
            self.add_event(ev)

    @property
    def event_count(self) -> int:
        """Number of events currently held in buffer."""
        return len(self._events)

    def get_events(self) -> List[TypingEvent]:
        """Return shallow copy of events."""
        return list(self._events)

    def to_dataframe(
        self,
        user_id: str = "live_user",
        session_id: str = "live_session",
    ) -> pd.DataFrame:
        """Convert buffered events to a DataFrame matching canonical event schemas.

        Args:
            user_id: Pseudonymous participant identifier.
            session_id: Session identifier.

        Returns:
            pd.DataFrame: Keystroke timing events with canonical columns.
        """
        if not self._events:
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "session_id",
                    "sequence_index",
                    "press_time",
                    "release_time",
                    "dwell_time",
                    "flight_time",
                    "pause_duration",
                    "typing_speed",
                    "backspace",
                    "error_flag",
                ]
            )

        records: List[Dict[str, Any]] = []
        for i, ev in enumerate(self._events):
            dwell = float(ev.dwell_time)
            flight = float(max(0.0, ev.flight_time)) if ev.flight_time is not None else np.nan
            pause_dur = flight if (pd.notnull(flight) and flight > self.pause_threshold_ms) else 0.0

            # Rolling typing speed estimation (WPM proxy based on recent key events)
            # 5 keystrokes ~= 1 word. Speed = (keys_in_window / 5) / (duration_minutes)
            window_start_idx = max(0, i - 10)
            window_events = self._events[window_start_idx : i + 1]
            if len(window_events) > 1:
                dt_ms = max(
                    100.0,
                    window_events[-1].release_timestamp
                    - window_events[0].press_timestamp,
                )
                minutes = dt_ms / 60000.0
                words = len(window_events) / 5.0
                wpm_est = float(np.clip(words / max(minutes, 0.001), 0.0, 250.0))
            else:
                wpm_est = 0.0

            is_bk = 1.0 if ev.is_backspace else 0.0
            is_err = 1.0 if (ev.is_backspace or ev.is_correction) else 0.0

            records.append({
                "user_id": user_id,
                "session_id": session_id,
                "sequence_index": ev.sequence_index,
                "press_time": ev.press_timestamp,
                "release_time": ev.release_timestamp,
                "dwell_time": dwell,
                "flight_time": flight,
                "pause_duration": pause_dur,
                "typing_speed": round(wpm_est, 2),
                "backspace": is_bk,
                "error_flag": is_err,
            })

        return pd.DataFrame(records)

    def extract_telemetry(self) -> Dict[str, Any]:
        """Compute real-time summary telemetry from buffered events.

        Returns:
            Dict[str, Any]: Live metrics for dashboard display.
        """
        count = len(self._events)
        if count == 0:
            return {
                "event_count": 0,
                "mean_dwell_ms": 0.0,
                "median_dwell_ms": 0.0,
                "mean_flight_ms": 0.0,
                "median_flight_ms": 0.0,
                "pause_count": 0,
                "pause_rate": 0.0,
                "estimated_wpm": 0.0,
                "backspace_count": 0,
                "correction_count": 0,
                "error_rate": 0.0,
                "duration_seconds": 0.0,
            }

        dwells = np.array([ev.dwell_time for ev in self._events if ev.dwell_time > 0.0])
        flights = np.array([
            ev.flight_time
            for ev in self._events
            if ev.flight_time is not None and ev.flight_time >= 0.0
        ])

        mean_dwell = float(np.mean(dwells)) if len(dwells) > 0 else 0.0
        median_dwell = float(np.median(dwells)) if len(dwells) > 0 else 0.0

        mean_flight = float(np.mean(flights)) if len(flights) > 0 else 0.0
        median_flight = float(np.median(flights)) if len(flights) > 0 else 0.0

        pause_count = int(np.sum(flights > self.pause_threshold_ms)) if len(flights) > 0 else 0
        pause_rate = float(pause_count / max(len(flights), 1))

        backspaces = sum(1 for ev in self._events if ev.is_backspace)
        corrections = sum(1 for ev in self._events if ev.is_correction)
        error_rate = float((backspaces + corrections) / max(count, 1))

        # Overall session duration and WPM
        t_start = self._events[0].press_timestamp
        t_end = self._events[-1].release_timestamp
        duration_sec = max(0.1, (t_end - t_start) / 1000.0)
        minutes = duration_sec / 60.0
        words = count / 5.0
        wpm = float(np.clip(words / max(minutes, 0.001), 0.0, 250.0))

        return {
            "event_count": count,
            "mean_dwell_ms": round(mean_dwell, 1),
            "median_dwell_ms": round(median_dwell, 1),
            "mean_flight_ms": round(mean_flight, 1),
            "median_flight_ms": round(median_flight, 1),
            "pause_count": pause_count,
            "pause_rate": round(pause_rate, 3),
            "estimated_wpm": round(wpm, 1),
            "backspace_count": backspaces,
            "correction_count": corrections,
            "error_rate": round(error_rate, 3),
            "duration_seconds": round(duration_sec, 2),
        }

    def generate_sequence_windows(
        self,
        sequence_length: Optional[int] = None,
        sequence_stride: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Slice buffered events into 3D temporal sequence arrays.

        Shape: (num_windows, sequence_length, 6)
        Features: [dwell_time, flight_time, pause_duration, typing_speed, backspace, error_flag]

        Args:
            sequence_length: Length of each window in keystroke events (default: self.sequence_length).
            sequence_stride: Step size between windows (default: self.sequence_stride).

        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]:
                - X: 3D sequence array of shape (num_windows, seq_len, 6)
                - window_metadata: List of window index ranges and timestamps.
        """
        seq_len = sequence_length or self.sequence_length
        stride = sequence_stride or self.sequence_stride
        num_features = len(CANONICAL_SEQUENCE_FEATURES)

        if len(self._events) < seq_len:
            return (
                np.empty((0, seq_len, num_features), dtype=np.float32),
                [],
            )

        df = self.to_dataframe()
        feature_matrix = df[CANONICAL_SEQUENCE_FEATURES].fillna(0.0).to_numpy(dtype=np.float32)

        windows: List[np.ndarray] = []
        meta: List[Dict[str, Any]] = []

        num_events = len(feature_matrix)
        for start_idx in range(0, num_events - seq_len + 1, stride):
            end_idx = start_idx + seq_len
            window = feature_matrix[start_idx:end_idx]
            windows.append(window)
            meta.append({
                "window_index": len(windows) - 1,
                "start_event_idx": start_idx,
                "end_event_idx": end_idx,
                "start_time_ms": float(df["press_time"].iloc[start_idx]),
                "end_time_ms": float(df["release_time"].iloc[end_idx - 1]),
            })

        X = np.array(windows, dtype=np.float32)
        return X, meta
