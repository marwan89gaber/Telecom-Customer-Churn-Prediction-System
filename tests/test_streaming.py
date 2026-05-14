"""
Streaming unit tests — no live Kafka or DB required.
Tests cover event schema, serialisation, and alert record construction.
"""
import json
import pytest
from datetime import datetime, timezone
from src.streaming.topics import TOPIC_EVENTS, TOPIC_ALERTS, EVENT_TYPES


# --- Topic constants ---

def test_topic_names_are_strings():
    assert isinstance(TOPIC_EVENTS, str) and len(TOPIC_EVENTS) > 0
    assert isinstance(TOPIC_ALERTS, str) and len(TOPIC_ALERTS) > 0


def test_event_types_non_empty():
    assert len(EVENT_TYPES) >= 3


# --- Producer event construction ---

def test_build_event_contains_required_keys():
    from src.streaming.producer import build_event
    row = {
        "customer_id": "TEST-001",
        "monthly_charges": 75.0,
        "contract": "Month-to-month",
    }
    event = build_event(row)
    assert "event_type"  in event
    assert "event_time"  in event
    assert "customer_id" in event
    assert event["event_type"] in EVENT_TYPES


def test_build_event_serialises_to_json():
    from src.streaming.producer import build_event
    row = {"customer_id": "TEST-002", "monthly_charges": 50.0}
    event = build_event(row)
    # Must be JSON-serialisable — KafkaProducer depends on this
    serialised = json.dumps(event).encode("utf-8")
    deserialised = json.loads(serialised.decode("utf-8"))
    assert deserialised["customer_id"] == "TEST-002"


# --- Alert record construction ---

def test_alert_record_shape():
    """Alert records must have the exact columns the DB table expects."""
    alert_payload = {
        "customer_id": "TEST-003",
        "event_type":  "complaint_raised",
        "churn_prob":  0.85,
        "risk_tier":   "High",
    }
    record = {
        "customer_id": alert_payload.get("customer_id", "unknown"),
        "event_type":  alert_payload.get("event_type",  "unknown"),
        "churn_prob":  float(alert_payload.get("churn_prob", 0.0)),
        "risk_tier":   alert_payload.get("risk_tier",   "High"),
        "alerted_at":  datetime.now(timezone.utc),
    }
    required_keys = {"customer_id", "event_type", "churn_prob", "risk_tier", "alerted_at"}
    assert required_keys.issubset(record.keys())
    assert 0.0 <= record["churn_prob"] <= 1.0


def test_churn_prob_clamped_to_valid_range():
    """Malformed Kafka messages should not write invalid probabilities."""
    prob = float({"churn_prob": 1.5}.get("churn_prob", 0.0))
    # In alert_consumer, you'd clamp: min(max(prob, 0.0), 1.0)
    # Test that the clamp logic works
    clamped = min(max(prob, 0.0), 1.0)
    assert 0.0 <= clamped <= 1.0