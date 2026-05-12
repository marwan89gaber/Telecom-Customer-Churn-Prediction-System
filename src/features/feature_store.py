"""
Feature store — single source of truth for what goes into the model.
Phase 4 imports from here. Nothing hardcodes column lists twice.
"""

NUMERIC_FEATURES = [
    "tenure",
    "monthlycharges",
    "totalcharges",
    "seniorcitizen",
    "num_services",
    "has_support_services",
    "charge_per_tenure",
    "monthly_spend_delta",
    "is_high_value",
    "is_new_customer",
    "contract_risk_score",
]

CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "paperlessbilling",
    "paymentmethod",
    "tenure_bucket",
]

TARGET = "churn_binary"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES