import os
from loguru import logger
import sys

def get_logger(name: str = "churn_etl"):
    os.makedirs("logs", exist_ok=True)
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> | {message}",
        level="INFO",
        colorize=True,
    )
    logger.add(
        f"logs/{name}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )
    return logger