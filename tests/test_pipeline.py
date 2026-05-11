import pandas as pd
import pytest
from src.pipeline.transform import transform

@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "customerID":    ["001", "002", "003", "002"],
        "gender":        ["Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 1],
        "Partner":       ["Yes", "No", "Yes", "No"],
        "Dependents":    ["No", "No", "Yes", "No"],
        "tenure":        [1, 0, 24, 0],
        "PhoneService":  ["Yes", "No", "Yes", "No"],
        "MultipleLines": ["No", "No phone service", "Yes", "No"],
        "InternetService":["DSL", "Fiber optic", "No", "Fiber optic"],
        "OnlineSecurity": ["No", "No", "Yes", "No"],
        "OnlineBackup":  ["Yes", "No", "No", "No"],
        "DeviceProtection":["No","No","Yes","No"],
        "TechSupport":   ["No", "No", "Yes", "No"],
        "StreamingTV":   ["No", "No", "No", "No"],
        "StreamingMovies":["No","No","No","No"],
        "Contract":      ["Month-to-month","Month-to-month","Two year","Month-to-month"],
        "PaperlessBilling":["Yes","Yes","No","Yes"],
        "PaymentMethod": ["Electronic check","Mailed check","Bank transfer (automatic)","Mailed check"],
        "MonthlyCharges":[29.85, 53.85, 42.30, 53.85],
        "TotalCharges":  ["29.85", " ", "1840.75", " "],
        "Churn":         ["No", "No", "No", "Yes"],
    })

def test_column_names_are_lowercase(raw_df):
    result = transform(raw_df)
    assert all(c == c.lower() for c in result.columns)

def test_total_charges_blank_becomes_zero(raw_df):
    result = transform(raw_df)
    assert result["totalcharges"].isna().sum() == 0
    assert (result["totalcharges"] == 0.0).any()

def test_duplicate_customer_ids_dropped(raw_df):
    result = transform(raw_df)
    assert result["customer_id"].duplicated().sum() == 0
    assert len(result) == 3  # 4 rows, 1 duplicate dropped

def test_churn_values_unchanged(raw_df):
    result = transform(raw_df)
    assert set(result["churn"].unique()).issubset({"Yes", "No"})