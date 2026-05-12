"""
Inference module — imported by Phase 5 (batch scoring) and Phase 7 (Flask API).
Single source of truth for how predictions are produced and risk tiers assigned.
"""
from pathlib import Path
import joblib
import pandas as pd
from src.features.engineer import engineer_features
from src.features.feature_store import ALL_FEATURES
from src.utils.logger import get_logger

logger = get_logger("predict")
MODEL_PATH = Path("models/churn_model.pkl")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python -m src.models.train` first."
        )
    return joblib.load(MODEL_PATH)


def get_risk_tier(prob: float) -> str:
    """
    Business-driven thresholds:
    - High  (≥0.70): escalate to retention team immediately
    - Medium (0.40–0.69): proactive outreach, targeted offer
    - Low   (<0.40): monitor, no intervention required
    """
    if prob >= 0.70:
        return "High"
    elif prob >= 0.40:
        return "Medium"
    return "Low"


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a clean customer DataFrame (transform.py output).
    Returns the same DataFrame with churn_prob and risk_tier appended.
    """
    model = load_model()
    df = engineer_features(df)
    probs = model.predict_proba(df[ALL_FEATURES])[:, 1]
    result = df.copy()
    result["churn_prob"] = probs.round(4)
    result["risk_tier"] = [get_risk_tier(p) for p in probs]
    logger.info(
        f"Scored {len(result):,} customers | "
        f"High: {(result['risk_tier']=='High').sum()} | "
        f"Medium: {(result['risk_tier']=='Medium').sum()} | "
        f"Low: {(result['risk_tier']=='Low').sum()}"
    )
    return result


def predict_single(customer: dict) -> dict:
    """
    Accepts a single customer record as a plain dict.
    Returns churn_prob and risk_tier.
    Called by the Flask API in Phase 7.
    """
    result = predict_batch(pd.DataFrame([customer]))
    prob = float(result["churn_prob"].iloc[0])
    return {
        "customer_id": customer.get("customer_id", "unknown"),
        "churn_prob":  prob,
        "risk_tier":   get_risk_tier(prob),
    }