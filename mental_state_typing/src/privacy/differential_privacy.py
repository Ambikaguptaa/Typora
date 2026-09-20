"""Differential Privacy Module.

Provides calibrated Laplace mechanism noise injection for aggregate research queries
(e.g., population mean typing speed, cohort pause frequencies) to prevent
membership inference or reconstruction attacks while maintaining analytical utility.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np

from src.config.settings import settings


def laplace_noise(scale: float, seed: Optional[int] = None) -> float:
    """Generate a single random sample from a zero-mean Laplace distribution.

    Args:
        scale: Diversity/scale parameter b = sensitivity / epsilon.
        seed: Optional RNG seed for deterministic testing.

    Returns:
        float: Noise value drawn from Laplace(0, scale).
    """
    if scale <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    return float(rng.laplace(loc=0.0, scale=scale))


def private_mean(
    values: Iterable[Union[int, float]],
    bounds: Tuple[float, float],
    epsilon: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute a differentially private mean using the Laplace mechanism.

    Values are clamped to [bounds[0], bounds[1]] to enforce bounded sensitivity:
        Sensitivity Delta = (upper_bound - lower_bound) / N

    Args:
        values: Numerical observations (e.g. dwell times, typing speeds).
        bounds: Tuple of (lower_bound, upper_bound) defining domain clipping bounds.
        epsilon: Privacy budget parameter (> 0). Defaults to settings.dp_epsilon.
        seed: Optional random seed for testing reproducibility.

    Returns:
        Dict[str, Any]: Detailed result dictionary containing private estimate and provenance.

    Raises:
        ValueError: If input is empty or epsilon <= 0.
    """
    arr = np.array(list(values), dtype=float)
    if len(arr) == 0:
        raise ValueError("Cannot compute differentially private mean on empty series.")

    lower, upper = bounds
    if lower >= upper:
        raise ValueError(f"Invalid clipping bounds: lower ({lower}) must be < upper ({upper}).")

    eff_eps = epsilon if epsilon is not None else settings.dp_epsilon
    if eff_eps <= 0:
        raise ValueError(f"Privacy budget epsilon must be positive, got {eff_eps}.")

    n = len(arr)
    # Clip to enforce bounded sensitivity
    clipped = np.clip(arr, lower, upper)
    true_mean = float(np.mean(clipped))

    # L1 sensitivity for mean: (upper - lower) / N
    sensitivity = (upper - lower) / float(n)
    scale = sensitivity / eff_eps

    noise = laplace_noise(scale, seed=seed)
    private_val = true_mean + noise

    return {
        "private_value": round(private_val, 4),
        "true_value": round(true_mean, 4),
        "epsilon": eff_eps,
        "sensitivity": round(sensitivity, 6),
        "noise_scale": round(scale, 6),
        "noise_added": round(noise, 6),
        "sample_size": n,
        "bounds": bounds,
        "is_differentially_private": True,
        "mechanism": "Laplace",
    }


def private_count(
    values: Iterable[Any],
    epsilon: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute a differentially private count using the Laplace mechanism.

    Sensitivity Delta = 1.0 (adding or removing one record alters the count by at most 1).

    Args:
        values: Collection of items to count.
        epsilon: Privacy budget parameter (> 0). Defaults to settings.dp_epsilon.
        seed: Optional random seed for testing reproducibility.

    Returns:
        Dict[str, Any]: Detailed result dictionary containing private count and provenance.
    """
    items = list(values)
    n = len(items)

    eff_eps = epsilon if epsilon is not None else settings.dp_epsilon
    if eff_eps <= 0:
        raise ValueError(f"Privacy budget epsilon must be positive, got {eff_eps}.")

    sensitivity = 1.0
    scale = sensitivity / eff_eps

    noise = laplace_noise(scale, seed=seed)
    private_val = max(0.0, float(n) + noise)

    return {
        "private_value": round(private_val, 2),
        "true_value": n,
        "epsilon": eff_eps,
        "sensitivity": sensitivity,
        "noise_scale": round(scale, 6),
        "noise_added": round(noise, 6),
        "sample_size": n,
        "is_differentially_private": True,
        "mechanism": "Laplace",
    }


def compute_privacy_budget(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Track and compute cumulative privacy budget under linear basic composition.

    Args:
        queries: List of DP query result dictionaries containing 'epsilon'.

    Returns:
        Dict[str, Any]: Composition summary including total epsilon consumed.
    """
    total_epsilon = sum(q.get("epsilon", 0.0) for q in queries)
    return {
        "total_epsilon": round(total_epsilon, 4),
        "query_count": len(queries),
        "composition_theorem": "Basic Sequential Composition (Sum)",
        "queries": queries,
    }
