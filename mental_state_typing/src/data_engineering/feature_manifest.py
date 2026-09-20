"""Feature Manifest Module.

Documents every engineered feature supported by the behavioral typing pipeline:
feature name, category, source columns, calculation method, unit, description,
and required raw columns.

Dynamically checks availability against candidate dataset schemas and generates
a structured manifest documenting both available and unavailable features.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class FeatureMetadata:
    """Metadata definition for an engineered behavioral feature."""

    feature_name: str
    category: str  # 'timing', 'pause', 'speed', 'variability', 'error_correction', 'word', 'session'
    source_columns: List[str]
    calculation_method: str
    unit: str  # 'ms', 'wpm', 'ratio', 'count', 'seconds', 'dimensionless'
    description: str
    availability: bool
    required_columns: List[str]
    reason_if_unavailable: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return asdict(self)


# Master Catalog of all possible features known to the pipeline
FEATURE_CATALOG: List[Dict[str, Any]] = [
    # 1. Dwell Time Features
    {
        "feature_name": "mean_dwell_time",
        "category": "timing",
        "source_columns": ["press_time", "release_time", "dwell_time", "hold_time"],
        "calculation_method": "Mean of (release_time - press_time) or dwell_time across observations",
        "unit": "ms",
        "description": "Average duration that keys remain depressed (dwell / hold latency)",
        "required_columns": [["dwell_time"], ["hold_time"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "median_dwell_time",
        "category": "timing",
        "source_columns": ["press_time", "release_time", "dwell_time", "hold_time"],
        "calculation_method": "50th percentile (median) of key hold durations",
        "unit": "ms",
        "description": "Median key hold duration (robust against outlier dwell spikes)",
        "required_columns": [["dwell_time"], ["hold_time"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "std_dwell_time",
        "category": "timing",
        "source_columns": ["press_time", "release_time", "dwell_time", "hold_time"],
        "calculation_method": "Standard deviation of key hold durations",
        "unit": "ms",
        "description": "Dispersion of key hold durations indicating motor consistency",
        "required_columns": [["dwell_time"], ["hold_time"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "min_dwell_time",
        "category": "timing",
        "source_columns": ["press_time", "release_time", "dwell_time", "hold_time"],
        "calculation_method": "Minimum valid key hold duration",
        "unit": "ms",
        "description": "Shortest measured key hold duration",
        "required_columns": [["dwell_time"], ["hold_time"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "max_dwell_time",
        "category": "timing",
        "source_columns": ["press_time", "release_time", "dwell_time", "hold_time"],
        "calculation_method": "Maximum valid key hold duration",
        "unit": "ms",
        "description": "Longest measured key hold duration",
        "required_columns": [["dwell_time"], ["hold_time"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "p25_dwell_time",
        "category": "timing",
        "source_columns": ["press_time", "release_time", "dwell_time", "hold_time"],
        "calculation_method": "25th percentile of key hold durations",
        "unit": "ms",
        "description": "First quartile of key hold duration",
        "required_columns": [["dwell_time"], ["hold_time"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "p75_dwell_time",
        "category": "timing",
        "source_columns": ["press_time", "release_time", "dwell_time", "hold_time"],
        "calculation_method": "75th percentile of key hold durations",
        "unit": "ms",
        "description": "Third quartile of key hold duration",
        "required_columns": [["dwell_time"], ["hold_time"], ["press_time", "release_time"]],
    },

    # 2. Flight / Inter-Key Interval Features
    {
        "feature_name": "mean_flight_time",
        "category": "timing",
        "source_columns": ["press_time", "release_time", "flight_time", "iki"],
        "calculation_method": "Mean transition time between consecutive key events (press_i - release_{i-1})",
        "unit": "ms",
        "description": "Average inter-key transition latency (flight time)",
        "required_columns": [["flight_time"], ["iki"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "median_flight_time",
        "category": "timing",
        "source_columns": ["press_time", "release_time", "flight_time", "iki"],
        "calculation_method": "50th percentile (median) of inter-key flight times",
        "unit": "ms",
        "description": "Median transition time between consecutive keystrokes",
        "required_columns": [["flight_time"], ["iki"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "std_flight_time",
        "category": "timing",
        "source_columns": ["press_time", "release_time", "flight_time", "iki"],
        "calculation_method": "Standard deviation of inter-key flight times",
        "unit": "ms",
        "description": "Rhythm stability/variability across consecutive keystrokes",
        "required_columns": [["flight_time"], ["iki"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "min_flight_time",
        "category": "timing",
        "source_columns": ["press_time", "release_time", "flight_time", "iki"],
        "calculation_method": "Minimum non-negative inter-key flight time",
        "unit": "ms",
        "description": "Shortest transition interval between consecutive key presses",
        "required_columns": [["flight_time"], ["iki"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "max_flight_time",
        "category": "timing",
        "source_columns": ["press_time", "release_time", "flight_time", "iki"],
        "calculation_method": "Maximum non-negative inter-key flight time",
        "unit": "ms",
        "description": "Longest transition interval between consecutive key presses",
        "required_columns": [["flight_time"], ["iki"], ["press_time", "release_time"]],
    },

    # 3. Pause Features
    {
        "feature_name": "pause_rate",
        "category": "pause",
        "source_columns": ["flight_time", "pause_duration", "press_time", "release_time"],
        "calculation_method": "Count of inter-key intervals exceeding pause threshold divided by total intervals",
        "unit": "ratio",
        "description": "Proportion of keystroke transitions that represent behavioral pauses (> threshold)",
        "required_columns": [["flight_time"], ["pause_duration"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "mean_pause_duration",
        "category": "pause",
        "source_columns": ["flight_time", "pause_duration"],
        "calculation_method": "Mean duration of intervals classified as pauses",
        "unit": "ms",
        "description": "Average duration of cognitive hesitation pauses",
        "required_columns": [["flight_time"], ["pause_duration"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "median_pause_duration",
        "category": "pause",
        "source_columns": ["flight_time", "pause_duration"],
        "calculation_method": "Median duration of intervals classified as pauses",
        "unit": "ms",
        "description": "Median duration of cognitive hesitation pauses",
        "required_columns": [["flight_time"], ["pause_duration"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "max_pause_duration",
        "category": "pause",
        "source_columns": ["flight_time", "pause_duration"],
        "calculation_method": "Maximum duration of pauses in observation window",
        "unit": "ms",
        "description": "Longest observed cognitive pause duration",
        "required_columns": [["flight_time"], ["pause_duration"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "long_pause_count",
        "category": "pause",
        "source_columns": ["flight_time", "pause_duration"],
        "calculation_method": "Number of intervals exceeding long pause threshold (e.g. 2.0s)",
        "unit": "count",
        "description": "Count of extended cognitive disengagements / long pauses",
        "required_columns": [["flight_time"], ["pause_duration"], ["press_time", "release_time"]],
    },

    # 4. Typing Speed Features
    {
        "feature_name": "mean_typing_speed_wpm",
        "category": "speed",
        "source_columns": ["typing_speed", "press_time", "release_time"],
        "calculation_method": "Mean words-per-minute (keystrokes / 5 / minutes or provided wpm field)",
        "unit": "wpm",
        "description": "Average typing throughput in words per minute",
        "required_columns": [["typing_speed"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "median_typing_speed_wpm",
        "category": "speed",
        "source_columns": ["typing_speed", "press_time", "release_time"],
        "calculation_method": "Median words-per-minute throughput",
        "unit": "wpm",
        "description": "Median typing speed across intervals",
        "required_columns": [["typing_speed"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "std_typing_speed_wpm",
        "category": "speed",
        "source_columns": ["typing_speed", "press_time", "release_time"],
        "calculation_method": "Standard deviation of typing speed across window",
        "unit": "wpm",
        "description": "Typing cadence pace variance",
        "required_columns": [["typing_speed"], ["press_time", "release_time"]],
    },

    # 5. Variability Features
    {
        "feature_name": "cv_dwell_time",
        "category": "variability",
        "source_columns": ["press_time", "release_time", "dwell_time", "hold_time"],
        "calculation_method": "std_dwell_time / max(mean_dwell_time, 1e-5)",
        "unit": "dimensionless",
        "description": "Coefficient of variation of dwell time (normalized motor variability)",
        "required_columns": [["dwell_time"], ["hold_time"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "cv_flight_time",
        "category": "variability",
        "source_columns": ["press_time", "release_time", "flight_time", "iki"],
        "calculation_method": "std_flight_time / max(mean_flight_time, 1e-5)",
        "unit": "dimensionless",
        "description": "Coefficient of variation of flight time (rhythm irregularity indicator)",
        "required_columns": [["flight_time"], ["iki"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "mad_dwell_time",
        "category": "variability",
        "source_columns": ["press_time", "release_time", "dwell_time", "hold_time"],
        "calculation_method": "Median absolute deviation of dwell time: median(|x - median(x)|)",
        "unit": "ms",
        "description": "Robust median absolute deviation of key hold durations",
        "required_columns": [["dwell_time"], ["hold_time"], ["press_time", "release_time"]],
    },
    {
        "feature_name": "mad_flight_time",
        "category": "variability",
        "source_columns": ["press_time", "release_time", "flight_time", "iki"],
        "calculation_method": "Median absolute deviation of flight time: median(|x - median(x)|)",
        "unit": "ms",
        "description": "Robust median absolute deviation of inter-key transitions",
        "required_columns": [["flight_time"], ["iki"], ["press_time", "release_time"]],
    },

    # 6. Error & Backspace Correction Features
    {
        "feature_name": "total_backspaces",
        "category": "error_correction",
        "source_columns": ["backspace", "key"],
        "calculation_method": "Count of backspace key events in session",
        "unit": "count",
        "description": "Total backspace occurrences indicating editing or correction activity",
        "required_columns": [["backspace"], ["key"]],
    },
    {
        "feature_name": "backspace_rate",
        "category": "error_correction",
        "source_columns": ["backspace", "key"],
        "calculation_method": "total_backspaces / max(keystroke_count, 1)",
        "unit": "ratio",
        "description": "Frequency of backspace corrections per total keystroke",
        "required_columns": [["backspace"], ["key"]],
    },
    {
        "feature_name": "total_errors",
        "category": "error_correction",
        "source_columns": ["error_flag", "is_error", "error"],
        "calculation_method": "Sum of validated error indicators in dataset",
        "unit": "count",
        "description": "Count of recorded typing errors (available only if dataset provides explicit error annotations)",
        "required_columns": [["error_flag"], ["is_error"], ["error"]],
    },
    {
        "feature_name": "error_rate",
        "category": "error_correction",
        "source_columns": ["error_flag", "is_error", "error"],
        "calculation_method": "total_errors / max(keystroke_count, 1)",
        "unit": "ratio",
        "description": "Error events per keystroke",
        "required_columns": [["error_flag"], ["is_error"], ["error"]],
    },

    # 7. Word Completion Features (Requires word boundary or word timestamps)
    {
        "feature_name": "mean_word_completion_time",
        "category": "word",
        "source_columns": ["word_time", "word_duration", "word_timestamp"],
        "calculation_method": "Mean elapsed time to complete discrete word tokens",
        "unit": "ms",
        "description": "Average duration required to type a complete word (requires word delimiter timestamps)",
        "required_columns": [["word_time"], ["word_duration"], ["word_timestamp"]],
    },
    {
        "feature_name": "word_completion_variability",
        "category": "word",
        "source_columns": ["word_time", "word_duration", "word_timestamp"],
        "calculation_method": "Standard deviation of word completion times",
        "unit": "ms",
        "description": "Variability in word completion intervals",
        "required_columns": [["word_time"], ["word_duration"], ["word_timestamp"]],
    },

    # 8. Session Level Aggregates
    {
        "feature_name": "session_duration_sec",
        "category": "session",
        "source_columns": ["press_time", "release_time", "timestamp"],
        "calculation_method": "(max(timestamp) - min(timestamp)) in seconds",
        "unit": "seconds",
        "description": "Total elapsed time of typing session",
        "required_columns": [["timestamp"], ["press_time"]],
    },
    {
        "feature_name": "keystroke_count",
        "category": "session",
        "source_columns": ["timestamp", "press_time", "key", "dwell_time"],
        "calculation_method": "Total number of observed keystroke events in session",
        "unit": "count",
        "description": "Total keystroke volume in session",
        "required_columns": [["timestamp"], ["press_time"], ["key"], ["dwell_time"], ["user_id"]],
    },
]


def _check_column_set_availability(
    available_cols: List[str], required_options: List[List[str]]
) -> Tuple[bool, List[str]]:
    """Determine if at least one combination of required columns is satisfied."""
    lowered_available = {c.lower().strip() for c in available_cols}
    for option in required_options:
        if all(req.lower().strip() in lowered_available for req in option):
            # Matched this valid option
            return True, option
    # Missing all valid options
    missing_desc = " OR ".join(["+".join(opt) for opt in required_options])
    return False, [missing_desc]


def generate_feature_manifest(
    available_columns: List[str],
) -> Dict[str, Any]:
    """Generate a structured feature manifest evaluating candidate features against available dataset columns.

    Args:
        available_columns: List of columns physically present in the target dataset.

    Returns:
        Dict[str, Any]: Structured manifest containing:
            - available_features: list of FeatureMetadata dicts for computable features
            - unavailable_features: list of FeatureMetadata dicts with missing dependency explanations
            - total_catalog_features: int
            - available_feature_count: int
            - unavailable_feature_count: int
    """
    available_list: List[Dict[str, Any]] = []
    unavailable_list: List[Dict[str, Any]] = []

    for entry in FEATURE_CATALOG:
        is_avail, matched_or_missing = _check_column_set_availability(
            available_columns, entry["required_columns"]
        )
        if is_avail:
            meta = FeatureMetadata(
                feature_name=entry["feature_name"],
                category=entry["category"],
                source_columns=matched_or_missing,
                calculation_method=entry["calculation_method"],
                unit=entry["unit"],
                description=entry["description"],
                availability=True,
                required_columns=matched_or_missing,
                reason_if_unavailable=None,
            )
            available_list.append(meta.to_dict())
        else:
            reason = (
                f"Missing required raw columns. Dataset lacks: {matched_or_missing[0]}. "
                f"Feature is omitted to prevent artificial/fabricated data generation."
            )
            meta = FeatureMetadata(
                feature_name=entry["feature_name"],
                category=entry["category"],
                source_columns=entry["source_columns"],
                calculation_method=entry["calculation_method"],
                unit=entry["unit"],
                description=entry["description"],
                availability=False,
                required_columns=matched_or_missing,
                reason_if_unavailable=reason,
            )
            unavailable_list.append(meta.to_dict())

    return {
        "total_catalog_features": len(FEATURE_CATALOG),
        "available_feature_count": len(available_list),
        "unavailable_feature_count": len(unavailable_list),
        "available_features": available_list,
        "unavailable_features": unavailable_list,
    }


def save_feature_manifest(manifest: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Serialize the feature manifest as a JSON artifact."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
