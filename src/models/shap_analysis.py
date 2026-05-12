"""
SHAP explainability — run after train.py.
Outputs two plots to docs/screenshots/ for the README and Power BI dashboard.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from src.features.engineer import engineer_features
from src.features.feature_store import ALL_FEATURES
from src.models.predict import load_model
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("shap")
SCREENSHOTS_DIR = Path("docs/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def run_shap_analysis() -> None:
    logger.info("Loading model and data for SHAP analysis")
    pipeline = load_model()
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier  = pipeline.named_steps["classifier"]

    df = pd.read_csv(Config.PROCESSED_DATA_PATH)
    df = engineer_features(df)
    X  = df[ALL_FEATURES]

    X_transformed  = preprocessor.transform(X)
    feature_names  = preprocessor.get_feature_names_out()

    logger.info("Computing SHAP values (LinearExplainer)...")

    if isinstance(classifier, LogisticRegression):
        explainer = shap.LinearExplainer(classifier, X_transformed)
    else:
        explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(X_transformed)

    # Some libraries return [class0_shap, class1_shap] — keep churn=1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # 1. Global beeswarm — shows which features drive predictions and in which direction
    shap.summary_plot(shap_values, X_transformed,
                      feature_names=feature_names,
                      show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: shap_summary.png")

    # 2. Waterfall for the single highest-risk customer
    churn_probs      = pipeline.predict_proba(X)[:, 1]
    high_risk_idx    = int(np.argmax(churn_probs))
    base_val = (explainer.expected_value[1]
                if isinstance(explainer.expected_value, list)
                else explainer.expected_value)

    explanation = shap.Explanation(
        values       = shap_values[high_risk_idx],
        base_values  = base_val,
        data         = X_transformed[high_risk_idx],
        feature_names= list(feature_names),
    )
    shap.plots.waterfall(explanation, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "shap_waterfall_high_risk.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: shap_waterfall_high_risk.png (customer index {high_risk_idx})")


if __name__ == "__main__":
    run_shap_analysis()