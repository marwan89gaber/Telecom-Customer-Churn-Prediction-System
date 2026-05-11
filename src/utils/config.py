import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    POSTGRES_USER     = os.getenv("POSTGRES_USER", "churn_admin")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "changeme")
    POSTGRES_HOST     = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT     = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB       = os.getenv("POSTGRES_DB", "churn_db")

    RAW_DATA_PATH = "data/external/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    PROCESSED_DATA_PATH = "data/processed/customers_clean.csv"

    @classmethod
    def get_db_url(cls) -> str:
        return (
            f"postgresql+psycopg2://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}"
            f"@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
        )

    @classmethod
    def get_jdbc_url(cls) -> str:
        return (
            f"jdbc:postgresql://{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
        )