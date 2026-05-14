"""
Spark Structured Streaming consumer.

Reads customer events from Kafka, scores each micro-batch using the
trained sklearn model via mapInPandas, and publishes high-risk alerts
to the churn-alerts Kafka topic.

Run (from project root, venv active, JAVA_HOME set):
    python -m src.streaming.spark_consumer
"""
import json
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, TimestampType,
)

from src.streaming.topics import TOPIC_EVENTS, TOPIC_ALERTS
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("spark_consumer")

# Kafka bootstrap inside Docker network (Spark runs inside Docker)
from src.utils.config import Config
KAFKA_INTERNAL = Config.KAFKA_BOOTSTRAP_SERVERS

KAFKA_PACKAGES = (
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
    "org.apache.kafka:kafka-clients:3.5.1"
)

# Schema mirrors the customers table + event fields added by the producer
EVENT_SCHEMA = StructType([
    StructField("customer_id",      StringType(),  True),
    StructField("gender",           StringType(),  True),
    StructField("senior_citizen",   IntegerType(), True),
    StructField("partner",          StringType(),  True),
    StructField("dependents",       StringType(),  True),
    StructField("tenure",           IntegerType(), True),
    StructField("phone_service",    StringType(),  True),
    StructField("multiple_lines",   StringType(),  True),
    StructField("internet_service", StringType(),  True),
    StructField("online_security",  StringType(),  True),
    StructField("online_backup",    StringType(),  True),
    StructField("device_protection",StringType(),  True),
    StructField("tech_support",     StringType(),  True),
    StructField("streaming_tv",     StringType(),  True),
    StructField("streaming_movies", StringType(),  True),
    StructField("contract",         StringType(),  True),
    StructField("paperless_billing",StringType(),  True),
    StructField("payment_method",   StringType(),  True),
    StructField("monthly_charges",  DoubleType(),  True),
    StructField("total_charges",    DoubleType(),  True),
    StructField("churn",            StringType(),  True),
    StructField("event_type",       StringType(),  True),
    StructField("event_time",       StringType(),  True),
])

# DB → feature name map (mirrors batch_predict.py)
DB_TO_FEATURE = {
    "senior_citizen":    "seniorcitizen",
    "phone_service":     "phoneservice",
    "multiple_lines":    "multiplelines",
    "internet_service":  "internetservice",
    "online_security":   "onlinesecurity",
    "online_backup":     "onlinebackup",
    "device_protection": "deviceprotection",
    "tech_support":      "techsupport",
    "streaming_tv":      "streamingtv",
    "streaming_movies":  "streamingmovies",
    "paperless_billing": "paperlessbilling",
    "payment_method":    "paymentmethod",
    "monthly_charges":   "monthlycharges",
    "total_charges":     "totalcharges",
}


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("ChurnStreamingConsumer")
        .config("spark.jars.packages", KAFKA_PACKAGES)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )


def score_batch(batch_df, batch_id: int) -> None:
    """
    Called by foreachBatch for every Spark micro-batch.
    Converts to Pandas, runs sklearn inference, writes alerts to Kafka.
    """
    if batch_df.isEmpty():
        return

    import pandas as pd
    from src.models.predict import predict_batch
    from src.pipeline.load import COLUMN_MAP
    from kafka import KafkaProducer

    pandas_df = batch_df.toPandas()

    # Rename DB columns → feature columns expected by engineer_features()
    pandas_df = pandas_df.rename(columns=DB_TO_FEATURE)

    # Score
    scored = predict_batch(pandas_df)

    # Filter to high-risk only
    high_risk = scored[scored["risk_tier"] == "High"]

    if high_risk.empty:
        logger.info(f"Batch {batch_id}: {len(scored)} events scored | 0 high-risk alerts")
        return

    logger.info(
        f"Batch {batch_id}: {len(scored)} events scored | "
        f"{len(high_risk)} HIGH-RISK alerts"
    )

    # Publish alerts to churn-alerts topic
    alert_producer = KafkaProducer(
        bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
    )

    for _, row in high_risk.iterrows():
        alert = {
            "customer_id": str(row.get("customer_id", "unknown")),
            "event_type":  str(row.get("event_type", "unknown")),
            "churn_prob":  float(row["churn_prob"]),
            "risk_tier":   str(row["risk_tier"]),
        }
        alert_producer.send(
            topic=TOPIC_ALERTS,
            key=alert["customer_id"],
            value=alert,
        )

    alert_producer.flush()
    alert_producer.close()


def run_consumer() -> None:
    logger.info("Starting Spark Structured Streaming consumer")
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_INTERNAL)
        .option("subscribe", TOPIC_EVENTS)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # Kafka delivers value as binary — parse JSON with the defined schema
    parsed = (
        raw_stream
        .select(
            F.from_json(
                F.col("value").cast("string"),
                EVENT_SCHEMA
            ).alias("data")
        )
        .select("data.*")
    )

    query = (
        parsed.writeStream
        .foreachBatch(score_batch)
        .option("checkpointLocation", "/tmp/churn_streaming_checkpoint")
        .trigger(processingTime="10 seconds")   # micro-batch every 10s
        .start()
    )

    logger.info("Streaming query started — waiting for events")
    query.awaitTermination()


if __name__ == "__main__":
    run_consumer()