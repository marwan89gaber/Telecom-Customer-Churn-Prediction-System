import os
from loguru import logger
import sys

_configured = False

def get_logger(name: str = "churn_etl"):
    global _configured
    os.makedirs("logs", exist_ok=True)
    if not _configured:
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
        _configured = True
    return logger