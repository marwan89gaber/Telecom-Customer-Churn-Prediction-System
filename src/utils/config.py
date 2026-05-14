import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    POSTGRES_USER     = os.getenv("POSTGRES_USER", "airflow")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")
    POSTGRES_HOST     = os.getenv("POSTGRES_HOST", "postgres")
    POSTGRES_PORT     = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB       = os.getenv("POSTGRES_DB", "airflow")
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
    KAFKA_TOPIC_EVENTS      = os.getenv("KAFKA_TOPIC_EVENTS", "customer-events")
    KAFKA_TOPIC_ALERTS      = os.getenv("KAFKA_TOPIC_ALERTS", "churn-alerts")

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