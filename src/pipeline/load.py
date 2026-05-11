import pandas as pd
from sqlalchemy import create_engine, text
from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger("load")

# Exact column mapping: DataFrame → customers table
COLUMN_MAP = {
    "customer_id":       "customer_id",
    "gender":            "gender",
    "seniorcitizen":     "senior_citizen",
    "partner":           "partner",
    "dependents":        "dependents",
    "tenure":            "tenure",
    "phoneservice":      "phone_service",
    "multiplelines":     "multiple_lines",
    "internetservice":   "internet_service",
    "onlinesecurity":    "online_security",
    "onlinebackup":      "online_backup",
    "deviceprotection":  "device_protection",
    "techsupport":       "tech_support",
    "streamingtv":       "streaming_tv",
    "streamingmovies":   "streaming_movies",
    "contract":          "contract",
    "paperlessbilling":  "paperless_billing",
    "paymentmethod":     "payment_method",
    "monthlycharges":    "monthly_charges",
    "totalcharges":      "total_charges",
    "churn":             "churn",
}

def load(df: pd.DataFrame) -> None:
    logger.info("Connecting to PostgreSQL")
    engine = create_engine(Config.get_db_url())

    # Rename DataFrame columns to match the DB schema
    df = df.rename(columns=COLUMN_MAP)
    db_cols = list(COLUMN_MAP.values())
    df = df[[c for c in db_cols if c in df.columns]]

    # Truncate before reload to avoid duplicate primary key errors on re-runs
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE customers RESTART IDENTITY CASCADE"))
        conn.commit()
        logger.info("Truncated customers table before reload")

    df.to_sql(
        name="customers",
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500,
    )
    logger.info(f"Loaded {len(df):,} rows into customers table")

    # Verify row count in DB matches what we loaded
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM customers"))
        db_count = result.scalar()
    logger.info(f"DB verification: {db_count:,} rows in customers table")

    if db_count != len(df):
        raise ValueError(f"Row count mismatch — loaded {len(df)}, DB has {db_count}")