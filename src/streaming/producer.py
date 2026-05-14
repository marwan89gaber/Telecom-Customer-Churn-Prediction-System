"""
Kafka producer — simulates a live CRM event stream.

Each message is a customer snapshot enriched with an event_type.
The producer reads from PostgreSQL and re-emits records continuously
at a configurable rate to simulate real-time activity.

Run:
    python -m src.streaming.producer
"""
import json
import random
import time
from datetime import datetime, timezone

import pandas as pd
from kafka import KafkaProducer
from sqlalchemy import create_engine

from src.streaming.topics import TOPIC_EVENTS, EVENT_TYPES
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("producer")

SEND_INTERVAL_SECONDS = 0.5   # one event every 500ms — adjust for demos
HIGH_RISK_BOOST       = 3     # high-risk customers appear 3x more often


def build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        acks="all",               # wait for broker acknowledgement
        retries=3,
    )


def load_customers() -> pd.DataFrame:
    engine = create_engine(Config.get_db_url())
    df = pd.read_sql("SELECT * FROM customers", con=engine)
    logger.info(f"Loaded {len(df):,} customers from PostgreSQL")
    return df


def build_event(row: dict) -> dict:
    """
    Attaches an event_type and timestamp to a customer record.
    This simulates a CRM system emitting a customer interaction event.
    """
    return {
        **row,
        "event_type": random.choice(EVENT_TYPES),
        "event_time": datetime.now(timezone.utc).isoformat(),
    }


def run_producer(limit: int = 0) -> None:
    """
    Streams customer events to Kafka.
    limit=0 means run forever. limit=N stops after N events (used in tests).
    """
    logger.info("Starting Kafka producer")
    producer = build_producer()
    customers = load_customers()

    # Weight high-charge customers more heavily to generate realistic alert volume
    customers["_weight"] = customers["monthly_charges"].clip(lower=1)
    weights = customers["_weight"] / customers["_weight"].sum()

    sent = 0
    try:
        while True:
            row = customers.sample(1, weights=weights).iloc[0].to_dict()
            # Remove internal weight column before sending
            row.pop("_weight", None)

            event = build_event(row)
            producer.send(
                topic=TOPIC_EVENTS,
                key=str(row["customer_id"]),
                value=event,
            )
            sent += 1
            logger.info(
                f"Sent [{sent}] customer_id={row['customer_id']} "
                f"event={event['event_type']}"
            )

            if limit and sent >= limit:
                logger.info(f"Reached limit of {limit} events — stopping")
                break

            time.sleep(SEND_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("Producer interrupted by user")
    finally:
        producer.flush()
        producer.close()
        logger.info(f"Producer closed — total events sent: {sent}")


if __name__ == "__main__":
    run_producer()