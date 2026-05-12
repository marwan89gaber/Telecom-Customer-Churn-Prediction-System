import pytest
import pandas as pd
from pathlib import Path
from src.models.predict import predict_single, predict_batch, get_risk_tier

SAMPLE_CUSTOMER = {
    "customer_id":    "TEST-001",
    "gender":         "Male",
    "seniorcitizen":  0,
    "partner":        "Yes",
    "dependents":     "No",
    "tenure":         2,
    "phoneservice":   "Yes",
    "multiplelines":  "No",
    "internetservice":"Fiber optic",
    "onlinesecurity": "No",
    "onlinebackup":   "No",
    "deviceprotection":"No",
    "techsupport":    "No",
    "streamingtv":    "No",
    "streamingmovies":"No",
    "contract":       "Month-to-month",
    "paperlessbilling":"Yes",
    "paymentmethod":  "Electronic check",
    "monthlycharges": 70.35,
    "totalcharges":   140.70,
    "churn":          "No",
}


@pytest.mark.skipif(
    not Path("models/churn_model.pkl").exists(),
    reason="Run `python -m src.models.train` first",
)
class TestModelInference:
    def test_predict_single_keys(self):
        result = predict_single(SAMPLE_CUSTOMER)
        assert {"customer_id", "churn_prob", "risk_tier"} <= result.keys()

    def test_churn_prob_is_valid_probability(self):
        result = predict_single(SAMPLE_CUSTOMER)
        assert 0.0 <= result["churn_prob"] <= 1.0

    def test_risk_tier_is_valid(self):
        result = predict_single(SAMPLE_CUSTOMER)
        assert result["risk_tier"] in {"High", "Medium", "Low"}

    def test_predict_batch_appends_columns(self):
        df = pd.DataFrame([SAMPLE_CUSTOMER, SAMPLE_CUSTOMER])
        result = predict_batch(df)
        assert "churn_prob" in result.columns
        assert "risk_tier" in result.columns
        assert len(result) == 2


# These run without the model — pure logic tests
def test_risk_tier_high_boundary():
    assert get_risk_tier(0.70) == "High"
    assert get_risk_tier(1.00) == "High"

def test_risk_tier_medium_boundary():
    assert get_risk_tier(0.40) == "Medium"
    assert get_risk_tier(0.69) == "Medium"

def test_risk_tier_low_boundary():
    assert get_risk_tier(0.39) == "Low"
    assert get_risk_tier(0.00) == "Low"