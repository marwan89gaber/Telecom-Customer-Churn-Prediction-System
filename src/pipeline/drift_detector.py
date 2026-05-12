"""
Drift detection using Population Stability Index (PSI).
Reference distribution is saved during training → models/training_distribution.json.
PSI thresholds (industry standard):
  < 0.10  — stable, no action needed
  0.10–0.20 — moderate shift, monitor closely
  ≥ 0.20  — significant drift, trigger retraining
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("drift_detector")

DIST_PATH    = Path("models/training_distribution.json")
PSI_WARNING  = 0.10
PSI_CRITICAL = 0.20


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute PSI between expected (training) and actual (current) distributions.
    Bin edges are always derived from the training data so they stay consistent
    across runs regardless of what the current data looks like.
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # sparse features can produce duplicate edges

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0]

    # Replace zeros to avoid log(0); 0.0001 is the conventional floor
    expected_pct = np.where(expected_counts == 0, 0.0001, expected_counts / len(expected))
    actual_pct   = np.where(actual_counts   == 0, 0.0001, actual_counts   / len(actual))

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 6)


def save_training_distribution(df: pd.DataFrame, features: list) -> None:
    """
    Called from train.py after training completes.
    Saves raw values for each numeric feature as the drift reference baseline.
    """
    dist = {}
    for feature in features:
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
            values = df[feature].dropna().tolist()
            dist[feature] = {
                "mean":   round(float(df[feature].mean()), 6),
                "std":    round(float(df[feature].std()),  6),
                "values": values,
            }
    DIST_PATH.parent.mkdir(exist_ok=True)
    DIST_PATH.write_text(json.dumps(dist, indent=2))
    logger.info(f"Training distribution saved → {DIST_PATH} ({len(dist)} features)")


def check_drift(current_df: pd.DataFrame) -> dict:
    """
    Computes PSI for every numeric feature against the training baseline.
    Returns a summary dict consumed by batch_predict.py and the Airflow DAG.
    """
    if not DIST_PATH.exists():
        raise FileNotFoundError(
            f"No training distribution at {DIST_PATH}. "
            "Run `python -m src.models.train` first."
        )

    reference  = json.loads(DIST_PATH.read_text())
    psi_scores = {}

    for feature, ref_data in reference.items():
        if feature not in current_df.columns:
            logger.warning(f"Feature '{feature}' missing from current data — skipping")
            continue

        actual = current_df[feature].dropna().values
        if len(actual) == 0:
            logger.warning(f"Feature '{feature}' has no data — skipping")
            continue

        expected = np.array(ref_data["values"])
        psi      = compute_psi(expected, actual)
        psi_scores[feature] = psi

        if psi >= PSI_CRITICAL:
            logger.warning(f"  {feature:30s} PSI={psi:.4f}  ← CRITICAL drift")
        elif psi >= PSI_WARNING:
            logger.warning(f"  {feature:30s} PSI={psi:.4f}  ← moderate drift")
        else:
            logger.info(   f"  {feature:30s} PSI={psi:.4f}  — stable")

    max_psi            = max(psi_scores.values()) if psi_scores else 0.0
    retraining_needed  = max_psi >= PSI_CRITICAL

    logger.info(f"Max PSI across all features: {max_psi:.4f}")
    logger.info(f"Retraining triggered: {retraining_needed}")
    return {
        "psi_scores":        psi_scores,
        "max_psi":           max_psi,
        "retraining_needed": retraining_needed,
    }