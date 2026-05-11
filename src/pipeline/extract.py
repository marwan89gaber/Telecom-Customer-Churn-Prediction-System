import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger("extract")

def extract() -> pd.DataFrame:
    logger.info(f"Reading raw data from: {Config.RAW_DATA_PATH}")
    try:
        df = pd.read_csv(Config.RAW_DATA_PATH)
        logger.info(f"Extracted {len(df):,} rows, {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        logger.error(f"Dataset not found at {Config.RAW_DATA_PATH}. Download it from Kaggle first.")
        raise