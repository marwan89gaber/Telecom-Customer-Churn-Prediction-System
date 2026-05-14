"""
Kafka topic constants and event schema.
Single source of truth — all producers and consumers import from here.
"""

TOPIC_EVENTS = "customer-events"
TOPIC_ALERTS = "churn-alerts"

# Event types that trigger a customer snapshot being sent to Kafka
EVENT_TYPES = [
    "complaint_raised",
    "usage_spike",
    "contract_query",
    "payment_late",
    "support_ticket",
    "downgrade_request",
]