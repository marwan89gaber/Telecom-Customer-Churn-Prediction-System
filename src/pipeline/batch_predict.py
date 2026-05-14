"""
Batch prediction pipeline — Phase 5.
Reads all customers from PostgreSQL, scores them with the trained model,
and writes results to the churn_predictions table.
Designed to be run on a schedule via cron or the Airflow DAG.
"""
import time
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import create_engine, text

from src.models.predict import predict_batch
from src.pipeline.drift_detector import check_drift
from src.pipeline.load import COLUMN_MAP
from src.features.engineer import engineer_features
from src.features.feature_store import NUMERIC_FEATURES
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("batch_predict")

# Reverse the ETL column map: DB column names → feature column names
DB_TO_FEATURE = {v: k for k, v in COLUMN_MAP.items()}


def read_customers(engine) -> pd.DataFrame:
    logger.info("Reading customers from PostgreSQL")
    df = pd.read_sql("SELECT * FROM customers", con=engine)
    logger.info(f"Loaded {len(df):,} customer records")
    return df


def rename_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    The customers table uses DB column names (senior_citizen, phone_service…).
    engineer_features() expects transform.py output names (seniorcitizen, phoneservice…).
    Reverse the COLUMN_MAP from load.py to translate between them.
    """
    return df.rename(columns=DB_TO_FEATURE)


def write_predictions(predictions: pd.DataFrame, engine) -> None:
    """
    Truncates the predictions table then bulk-inserts fresh results.
    Truncate-before-insert avoids duplicates on re-runs and keeps the
    table as a current snapshot rather than a growing history.
    """
    output = predictions[["customer_id", "churn_prob", "risk_tier"]].copy()
    output["predicted_at"] = datetime.now(timezone.utc)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM churn_predictions"))

    output.to_sql(
        "churn_predictions",
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500,
    )
    logger.info(f"Wrote {len(output):,} predictions to churn_predictions table")


def log_risk_summary(df: pd.DataFrame) -> None:
    total = len(df)
    for tier in ["High", "Medium", "Low"]:
        count = (df["risk_tier"] == tier).sum()
        logger.info(f"  {tier:6s}: {count:,}  ({count / total:.1%})")


def run_batch_predict() -> None:
    start = time.time()
    logger.info("=" * 60)
    logger.info("Phase 5 — Batch prediction started")

    engine = create_engine(Config.get_db_url())
    raw_df     = read_customers(engine)
    feature_df = rename_for_inference(raw_df)

    # Engineer once — reuse for both drift check and scoring
    from src.features.engineer import engineer_features
    from src.models.predict import load_model, get_risk_tier
    from src.features.feature_store import ALL_FEATURES

    engineered = engineer_features(feature_df)

    logger.info("Running drift check...")
    drift_result = check_drift(engineered[NUMERIC_FEATURES])
    if drift_result["retraining_needed"]:
        logger.warning("Significant feature drift detected (PSI ≥ 0.20).")

    # Score using already-engineered DataFrame — no second engineer_features call
    model = load_model()
    probs = model.predict_proba(engineered[ALL_FEATURES])[:, 1]
    scored_df = engineered.copy()
    scored_df["churn_prob"] = probs.round(4)
    scored_df["risk_tier"]  = [get_risk_tier(p) for p in probs]

    logger.info("Risk tier breakdown:")
    log_risk_summary(scored_df)
    write_predictions(scored_df, engine)

    elapsed = round(time.time() - start, 2)
    logger.info(f"Batch prediction complete in {elapsed}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_batch_predict()