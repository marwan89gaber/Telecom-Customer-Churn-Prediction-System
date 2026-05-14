"""
Alert consumer — reads high-risk alerts from Kafka and writes to PostgreSQL.

This is intentionally simple (no Spark): it's a persistence layer,
not a processing layer. All ML inference happened in spark_consumer.py.

Run:
    python -m src.streaming.alert_consumer
"""
import json
from datetime import datetime, timezone

from kafka import KafkaConsumer
from sqlalchemy import create_engine, text

from src.streaming.topics import TOPIC_ALERTS
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("alert_consumer")

INSERT_SQL = text("""
    INSERT INTO churn_alerts (customer_id, event_type, churn_prob, risk_tier, alerted_at)
    VALUES (:customer_id, :event_type, :churn_prob, :risk_tier, :alerted_at)
""")


def build_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        TOPIC_ALERTS,
        bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        group_id="churn-alert-persister",
        auto_offset_reset="earliest",   # on first start, process all existing alerts
        enable_auto_commit=True,
        consumer_timeout_ms=30_000,     # stop after 30s of silence (useful for tests)
    )


def run_alert_consumer(limit: int = 0) -> int:
    """
    Consumes alerts and writes to DB.
    limit=0 means run until timeout. limit=N stops after N records (for tests).
    Returns count of alerts written.
    """
    logger.info("Alert consumer starting")
    engine  = create_engine(Config.get_db_url())
    consumer = build_consumer()
    written = 0

    try:
        for message in consumer:
            alert = message.value
            record = {
                "customer_id": alert.get("customer_id", "unknown"),
                "event_type":  alert.get("event_type",  "unknown"),
                "churn_prob":  float(alert.get("churn_prob", 0.0)),
                "risk_tier":   alert.get("risk_tier",   "High"),
                "alerted_at":  datetime.now(timezone.utc),
            }
            with engine.connect() as conn:
                conn.execute(INSERT_SQL, record)
                conn.commit()

            written += 1
            logger.info(
                f"Alert written [{written}] | customer={record['customer_id']} "
                f"prob={record['churn_prob']:.3f} event={record['event_type']}"
            )

            if limit and written >= limit:
                logger.info(f"Reached limit of {limit} — stopping")
                break

    except Exception as e:
        logger.error(f"Alert consumer error: {e}")
        raise
    finally:
        consumer.close()
        logger.info(f"Alert consumer closed — {written} alerts written")

    return written


if __name__ == "__main__":
    run_alert_consumer()