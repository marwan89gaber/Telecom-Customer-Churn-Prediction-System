from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from sklearn.pipeline import Pipeline
from src.utils.logger import get_logger

logger = get_logger("evaluate")
SCREENSHOTS_DIR = Path("docs/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(pipeline: Pipeline,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "roc_auc":         round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision":   round(average_precision_score(y_test, y_prob), 4),
        "precision_churn": round(report["1"]["precision"], 4),
        "recall_churn":    round(report["1"]["recall"], 4),
        "f1_churn":        round(report["1"]["f1-score"], 4),
        "accuracy":        round(report["accuracy"], 4),
    }
    logger.info("Test set results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")
    return metrics


def save_evaluation_plots(pipeline: Pipeline,
                          X_test: pd.DataFrame,
                          y_test: pd.Series) -> None:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    logger.info("Evaluation plots saved to docs/screenshots/")