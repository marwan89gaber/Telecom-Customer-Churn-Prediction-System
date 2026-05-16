"""
Airflow DAG — daily churn batch prediction with drift-gated retraining.

To run locally with Airflow:
  pip install apache-airflow
  export AIRFLOW_HOME=$(pwd)/airflow
  airflow db init
  airflow scheduler &
  airflow webserver

Place this file (or symlink it) in $AIRFLOW_HOME/dags/.

DAG flow:
  check_drift
      ├── (drift critical) → retrain_model ──┐
      └── (drift stable)   → skip_retrain  ──┴→ batch_predict
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner":            "data-science",
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="churn_batch_prediction",
    description="Daily batch churn scoring with PSI drift detection",
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 2 * * *",   # 02:00 UTC daily
    catchup=False,
    default_args=default_args,
    tags=["churn", "ml", "batch"],
) as dag:

    def _check_drift(**context):
        import pandas as pd
        from sqlalchemy import create_engine
        from src.utils.config import Config
        from src.pipeline.batch_predict import read_customers, rename_for_inference
        from src.pipeline.drift_detector import check_drift
        from src.features.engineer import engineer_features
        from src.features.feature_store import NUMERIC_FEATURES

        engine = create_engine(Config.get_db_url())
        raw_df = read_customers(engine)
        feature_df = rename_for_inference(raw_df)
        engineered = engineer_features(feature_df)

        result = check_drift(engineered[NUMERIC_FEATURES])
        context["ti"].xcom_push(key="drift_result", value={
            "max_psi": float(result["max_psi"]),
            "retraining_needed": bool(result["retraining_needed"]),
        })

        return "retrain_model" if result["retraining_needed"] else "skip_retrain"

    def _retrain(**_):
        from src.models.train import run_training
        run_training()

    def _batch_predict(**_):
        from src.pipeline.batch_predict import run_batch_predict
        run_batch_predict()

    check_drift_task = BranchPythonOperator(
        task_id="check_drift",
        python_callable=_check_drift,
    )

    retrain_task = PythonOperator(
        task_id="retrain_model",
        python_callable=_retrain,
    )

    skip_retrain = EmptyOperator(task_id="skip_retrain")

    predict_task = PythonOperator(
        task_id="batch_predict",
        python_callable=_batch_predict,
        trigger_rule="none_failed_min_one_success",  # runs after either branch
    )

    check_drift_task >> [retrain_task, skip_retrain] >> predict_task