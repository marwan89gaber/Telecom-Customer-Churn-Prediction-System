"""
Kafka topic constants and event schema.
Single source of truth — all producers and consumers import from here.
"""
from src.utils.config import Config

TOPIC_EVENTS = Config.KAFKA_TOPIC_EVENTS
TOPIC_ALERTS = Config.KAFKA_TOPIC_ALERTS

# Event types that trigger a customer snapshot being sent to Kafka
EVENT_TYPES = [
    "complaint_raised",
    "usage_spike",
    "contract_query",
    "payment_late",
    "support_ticket",
    "downgrade_request",
]