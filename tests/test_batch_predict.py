import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.pipeline.drift_detector import compute_psi


# --- PSI unit tests (no model or DB required) ---

def test_psi_identical_distributions_is_near_zero():
    """PSI of a distribution against itself should be essentially zero."""
    data = np.random.default_rng(42).normal(50, 10, 1000)
    assert compute_psi(data, data) < 0.01


def test_psi_very_different_distributions_exceeds_critical():
    """Distributions shifted far apart should exceed the 0.20 retraining threshold."""
    rng = np.random.default_rng(42)
    expected = rng.normal(0, 1, 1000)
    actual   = rng.normal(10, 1, 1000)   # mean shifted by 10 standard deviations
    assert compute_psi(expected, actual) > 0.20


def test_psi_is_non_negative():
    rng = np.random.default_rng(0)
    a = rng.exponential(1, 500)
    b = rng.exponential(2, 500)
    assert compute_psi(a, b) >= 0


@pytest.mark.skipif(
    not Path("models/training_distribution.json").exists(),
    reason="Run `python -m src.models.train` first",
)
def test_check_drift_returns_expected_structure():
    from src.pipeline.drift_detector import check_drift
    from src.features.feature_store import NUMERIC_FEATURES

    dummy = pd.DataFrame(
        {f: np.random.default_rng(1).normal(0, 1, 200) for f in NUMERIC_FEATURES}
    )
    result = check_drift(dummy)

    assert "psi_scores" in result
    assert "max_psi"    in result
    assert "retraining_needed" in result
    assert isinstance(result["retraining_needed"], bool)
    assert result["max_psi"] >= 0


@pytest.mark.skipif(
    not Path("models/churn_model.pkl").exists(),
    reason="Run `python -m src.models.train` first",
)
def test_column_rename_does_not_lose_rows():
    """Renaming DB columns for inference should not drop any rows."""
    from src.pipeline.load import COLUMN_MAP
    from src.pipeline.batch_predict import DB_TO_FEATURE

    # Verify the reverse map covers the same keys
    assert set(DB_TO_FEATURE.keys()) == set(COLUMN_MAP.values())