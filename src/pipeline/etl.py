import time
from src.pipeline.extract import extract
from src.pipeline.transform import transform
from src.pipeline.load import load
from src.utils.logger import get_logger

logger = get_logger("etl")

def run_etl() -> None:
    logger.info("=" * 50)
    logger.info("ETL pipeline started")
    start = time.time()

    try:
        raw_df = extract()
        clean_df = transform(raw_df)
        load(clean_df)
    except Exception as e:
        logger.error(f"ETL failed: {e}")
        raise

    elapsed = round(time.time() - start, 2)
    logger.info(f"ETL pipeline completed successfully in {elapsed}s")
    logger.info("=" * 50)

if __name__ == "__main__":
    run_etl()