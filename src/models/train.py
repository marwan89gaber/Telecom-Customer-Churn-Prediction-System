import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.engineer import engineer_features
from src.features.feature_store import ALL_FEATURES, TARGET
from src.models.preprocessor import build_preprocessor
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("train")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = ["roc_auc", "f1", "precision", "recall"]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Loading data from {Config.PROCESSED_DATA_PATH}")
    df = pd.read_csv(Config.PROCESSED_DATA_PATH)
    df = engineer_features(df)
    X = df[ALL_FEATURES]
    y = df[TARGET]
    logger.info(f"Dataset: {X.shape[0]:,} rows | Churn rate: {y.mean():.1%}")
    return X, y


def get_candidates(scale_pos_weight: float) -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight, n_estimators=200,
            random_state=42, eval_metric="logloss", verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            is_unbalance=True, n_estimators=200,
            random_state=42, verbose=-1,
        ),
    }


def compare_models(X_train, y_train, scale_pos_weight: float) -> str:
    logger.info("Cross-validating all candidate models...")
    candidates = get_candidates(scale_pos_weight)
    results = {}

    for name, clf in candidates.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", clf),
        ])
        scores = cross_validate(pipe, X_train, y_train, cv=CV,
                                scoring=SCORING, n_jobs=-1)
        results[name] = {
            "roc_auc":   scores["test_roc_auc"].mean(),
            "f1":        scores["test_f1"].mean(),
            "precision": scores["test_precision"].mean(),
            "recall":    scores["test_recall"].mean(),
        }
        logger.info(
            f"{name:22s} | AUC: {results[name]['roc_auc']:.4f} | "
            f"F1: {results[name]['f1']:.4f} | "
            f"Recall: {results[name]['recall']:.4f}"
        )

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    logger.info(f"Winner (by AUC): {best_name}")
    return best_name


PARAM_GRIDS = {
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__solver": ["lbfgs", "liblinear"],
    },
    "Random Forest": {
        "classifier__n_estimators": [200, 300],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 4],
    },
    "XGBoost": {
        "classifier__n_estimators": [200, 300],
        "classifier__max_depth": [3, 4, 5],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__subsample": [0.8, 1.0],
    },
    "LightGBM": {
        "classifier__n_estimators": [200, 300],
        "classifier__max_depth": [4, 6, -1],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__num_leaves": [31, 63],
    },
}


def tune_model(X_train, y_train, best_name: str,
               scale_pos_weight: float) -> Pipeline:
    logger.info(f"Tuning {best_name} with GridSearchCV...")
    clf = get_candidates(scale_pos_weight)[best_name]
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", clf),
    ])
    grid = GridSearchCV(
        pipe, PARAM_GRIDS[best_name],
        cv=CV, scoring="roc_auc", n_jobs=-1, verbose=1,
    )
    grid.fit(X_train, y_train)
    logger.info(f"Best params: {grid.best_params_}")
    logger.info(f"Best CV AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


def run_training() -> None:
    start = time.time()
    logger.info("=" * 60)
    logger.info("Phase 4 — Training started")

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = round(neg / pos, 2)
    logger.info(f"Class ratio neg/pos: {scale_pos_weight}")

    best_name = compare_models(X_train, y_train, scale_pos_weight)
    best_pipeline = tune_model(X_train, y_train, best_name, scale_pos_weight)

    from src.models.evaluate import evaluate_model, save_evaluation_plots
    metrics = evaluate_model(best_pipeline, X_test, y_test)
    save_evaluation_plots(best_pipeline, X_test, y_test)

    model_path = MODELS_DIR / "churn_model.pkl"
    joblib.dump(best_pipeline, model_path)
    logger.info(f"Model saved → {model_path}")

    metadata = {
        "model_name": best_name,
        "trained_at": datetime.utcnow().isoformat(),
        "test_metrics": metrics,
        "features": ALL_FEATURES,
        "target": TARGET,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "churn_rate": float(y.mean()),
    }
    meta_path = MODELS_DIR / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Metadata saved → {meta_path}")

    logger.info(f"Phase 4 complete in {round(time.time() - start, 2)}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_training()