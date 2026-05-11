from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger("spark_batch")

JDBC_JAR = "docker/jars/postgresql-42.7.3.jar"

def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("TelecomChurnETL")
        .config("spark.jars", JDBC_JAR)
        .config("spark.sql.shuffle.partitions", "4")  # small dataset — don't over-partition
        .getOrCreate()
    )

def spark_etl() -> None:
    logger.info("Spark ETL started")
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    # Extract
    df = spark.read.csv(Config.RAW_DATA_PATH, header=True, inferSchema=True)
    logger.info(f"Spark read {df.count():,} rows")

    # Transform — mirrors the Pandas logic
    # Standardise column names
    for col in df.columns:
        new_name = (
            col.strip().lower()
            .replace(" ", "_")
            .replace("customerid", "customer_id")
        )
        if new_name != col:
            df = df.withColumnRenamed(col, new_name)

    # Fix TotalCharges (blank strings → 0.0)
    df = df.withColumn(
        "totalcharges",
        F.when(F.trim(F.col("totalcharges")) == "", "0")
         .otherwise(F.col("totalcharges"))
         .cast(DoubleType())
    )

    # Drop duplicates
    df = df.dropDuplicates(["customer_id"])

    # Log churn distribution
    logger.info("Churn distribution:")
    df.groupBy("churn").count().show()

    # Load → PostgreSQL via JDBC
    jdbc_props = {
        "user":     Config.POSTGRES_USER,
        "password": Config.POSTGRES_PASSWORD,
        "driver":   "org.postgresql.Driver",
    }
    df.write.jdbc(
        url=Config.get_jdbc_url(),
        table="customers",
        mode="overwrite",
        properties=jdbc_props,
    )
    logger.info("Spark ETL: data written to PostgreSQL via JDBC")
    spark.stop()

if __name__ == "__main__":
    spark_etl()