import pandas as pd
import os
from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger("transform")

def transform(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting transformation pipeline")
    df = df.copy()

    # 1. Standardise column names to snake_case
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    df.rename(columns={"customerid": "customer_id"}, inplace=True)
    logger.info("Column names standardised")

    # 2. Fix TotalCharges — IBM dataset stores blanks as " " for new customers
    initial_nulls = df["totalcharges"].isna().sum()
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    coerced = df["totalcharges"].isna().sum() - initial_nulls
    df["totalcharges"].fillna(0.0, inplace=True)
    logger.info(f"TotalCharges: coerced {coerced} blank strings → 0.0")

    # 3. Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # 4. Validate no duplicate customer IDs
    dupes = df["customer_id"].duplicated().sum()
    if dupes > 0:
        logger.warning(f"Found {dupes} duplicate customer IDs — dropping duplicates, keeping first")
        df.drop_duplicates(subset="customer_id", keep="first", inplace=True)
    else:
        logger.info("No duplicate customer IDs found")

    # 5. Validate expected churn values
    valid_churn = {"Yes", "No"}
    bad_churn = ~df["churn"].isin(valid_churn)
    if bad_churn.any():
        logger.warning(f"{bad_churn.sum()} rows with unexpected Churn values: {df.loc[bad_churn, 'churn'].unique()}")

    # 6. Log class distribution — critical for Phase 4 imbalance decisions
    churn_dist = df["churn"].value_counts(normalize=True).mul(100).round(2)
    logger.info(f"Churn distribution: {churn_dist.to_dict()}")

    # 7. Save processed copy for reference
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(Config.PROCESSED_DATA_PATH, index=False)
    logger.info(f"Processed data saved to {Config.PROCESSED_DATA_PATH}")

    logger.info(f"Transform complete — {len(df):,} clean rows ready")
    return df