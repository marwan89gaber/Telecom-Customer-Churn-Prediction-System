import pandas as pd
import pytest
from src.features.engineer import engineer_features, get_feature_columns

@pytest.fixture
def clean_df():
    """Minimal clean DataFrame that mirrors transform.py output."""
    return pd.DataFrame({
        "customer_id":    ["001", "002", "003"],
        "gender":         ["Male", "Female", "Male"],
        "seniorcitizen":  [0, 1, 0],
        "partner":        ["Yes", "No", "Yes"],
        "dependents":     ["No", "No", "Yes"],
        "tenure":         [1, 36, 60],
        "phoneservice":   ["Yes", "Yes", "No"],
        "multiplelines":  ["No", "Yes", "No phone service"],
        "internetservice":["DSL", "Fiber optic", "No"],
        "onlinesecurity": ["No", "Yes", "No internet service"],
        "onlinebackup":   ["Yes", "No", "No internet service"],
        "deviceprotection":["No","Yes","No internet service"],
        "techsupport":    ["No", "Yes", "No internet service"],
        "streamingtv":    ["No", "No", "No internet service"],
        "streamingmovies":["No", "Yes", "No internet service"],
        "contract":       ["Month-to-month", "One year", "Two year"],
        "paperlessbilling":["Yes", "Yes", "No"],
        "paymentmethod":  ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "monthlycharges": [29.85, 79.65, 20.25],
        "totalcharges":   [29.85, 2864.75, 1215.0],
        "churn":          ["No", "No", "No"],
    })

def test_tenure_bucket_created(clean_df):
    result = engineer_features(clean_df)
    assert "tenure_bucket" in result.columns
    assert result.loc[result["tenure"] == 1, "tenure_bucket"].iloc[0] == "new"
    assert result.loc[result["tenure"] == 60, "tenure_bucket"].iloc[0] == "loyal"

def test_num_services_range(clean_df):
    result = engineer_features(clean_df)
    assert result["num_services"].between(0, 9).all()

def test_is_new_customer_flag(clean_df):
    result = engineer_features(clean_df)
    assert result.loc[result["tenure"] == 1, "is_new_customer"].iloc[0] == 1
    assert result.loc[result["tenure"] == 60, "is_new_customer"].iloc[0] == 0

def test_contract_risk_score_values(clean_df):
    result = engineer_features(clean_df)
    assert set(result["contract_risk_score"].unique()).issubset({0, 1, 2})
    assert result.loc[result["contract"] == "Month-to-month", "contract_risk_score"].iloc[0] == 2

def test_no_nulls_in_engineered_columns(clean_df):
    result = engineer_features(clean_df)
    new_cols = [
        "tenure_bucket", "num_services", "has_support_services",
        "charge_per_tenure", "is_high_value", "is_new_customer",
        "contract_risk_score",
    ]
    assert result[new_cols].isnull().sum().sum() == 0

def test_feature_column_list_is_complete(clean_df):
    result = engineer_features(clean_df)
    feature_cols = get_feature_columns()
    missing = [c for c in feature_cols if c not in result.columns]
    assert missing == [], f"Missing columns: {missing}"