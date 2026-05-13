# Telecom Customer Churn Prediction Platform

End-to-end ML system for predicting telecom customer churn — covering
data engineering, batch + real-time scoring, REST API deployment,
and a business intelligence dashboard. Built to simulate a production
environment.

---

## Tech stack

Python · PySpark · Apache Kafka · PostgreSQL · Flask · Docker · Power BI · SHAP

---

## Quick start

**Prerequisites:** Python 3.10+, Docker Desktop, Java 11+ (for Spark jobs)

```bash
git clone https://github.com/marwan89gaber/Telecom-Customer-Churn-Prediction-System
cd telecom-churn-platform

# Environment variables
cp .env.example .env

# Python virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt

# Start PostgreSQL (init.sql runs automatically on first launch)
docker-compose up -d

# If re-running on an existing volume and tables are missing, reset with:
docker-compose down -v && docker-compose up -d

# Download dataset from Kaggle and place at:
# data/external/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## Running the pipeline

All scripts must be run as modules from the project root with the venv active.

```bash
# 1. Run the full ETL pipeline (extract → clean → transform → load into PostgreSQL)
python -m src.pipeline.etl

# 2. Verify data landed in the database
docker exec -it churn_postgres psql -U churn_admin -d churn_db \
  -c "SELECT churn, COUNT(*) FROM customers GROUP BY churn;"

# 3. Run the PySpark batch job (requires Java 11+ and JAVA_HOME set)
python -m src.pipeline.spark_batch

# 4. Run the EDA and feature engineering notebook
jupyter notebook notebooks/01_eda.ipynb

# 5. Train the model — expect 3–8 minutes with GridSearchCV
python -m src.models.train

# 6. Generate SHAP explainability plots (requires trained model)
python -m src.models.shap_analysis

# 7. Run all tests
pytest

# 8. Run the batch prediction pipeline (requires Docker PostgreSQL running)
python -m src.pipeline.batch_predict

# 9. Verify predictions in the database
docker exec -it churn_postgres psql -U churn_admin -d churn_db -c "SELECT risk_tier, COUNT(*) FROM churn_predictions GROUP BY risk_tier;"
```

---

## Project phases

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Foundation & setup | ✅ Done |
| 2 | Data engineering pipeline | ✅ Done |
| 3 | EDA & feature engineering | ✅ Done |
| 4 | ML modelling & evaluation | ✅ Done |
| 5 | Batch prediction pipeline | ✅ Done |
| 6 | Real-time streaming (Kafka) | ⏳ Planned |
| 7 | API, dashboard & deployment | ⏳ Planned |
| 8 | Documentation & packaging | ⏳ Planned |

---

## Repository layout

```text
telecom-churn-platform/
├── api/                  # Flask REST API (Phase 7)
├── dashboard/            # Power BI dashboards (Phase 7)
├── data/
│   ├── external/         # IBM Telco Churn dataset (not committed — see Dataset section)
│   ├── processed/        # cleaned datasets (not committed)
│   └── raw/              # original CSVs (not committed)
├── docker/
│   ├── jars/             # PostgreSQL JDBC driver (not committed — see Spark section)
│   └── init.sql          # DB schema bootstrap
├── docs/
│   ├── architecture/     # system architecture diagrams (Phase 7)
│   └── screenshots/      # dashboard and API screenshots (Phase 7)
├── models/               # trained model artifacts (Phase 4)
├── notebooks/
│   └── 01_eda.ipynb      # exploratory data analysis and feature engineering
├── airflow/
│   └── dags/
│       └── churn_batch_dag.py  # Airflow DAG for scheduled scoring + retraining
├── src/
│   ├── features/
│   │   ├── engineer.py   # 8 business-driven engineered features
│   │   └── feature_store.py  # single source of truth for feature columns
│   ├── models/           # model training and evaluation (Phase 4)
│   ├── pipeline/
│   │   ├── etl.py        # pipeline orchestrator
│   │   ├── extract.py    # CSV ingestion
│   │   ├── transform.py  # cleaning and standardisation
│   │   ├── load.py       # PostgreSQL loader
│   │   ├── spark_batch.py  # PySpark ETL job
│   │   ├── batch_predict.py  # batch scoring from DB → predictions table
│   │   └── drift_detector.py # PSI-based feature drift detection
│   ├── streaming/        # Kafka producers and consumers (Phase 6)
│   └── utils/
│       ├── config.py     # environment and path configuration
│       └── logger.py     # structured logging with loguru
├── tests/
│   ├── test_pipeline.py  # ETL transform unit tests
│   └── test_features.py  # feature engineering unit tests
│   ├── test_model.py         # model inference and risk tier tests
│   └── test_batch_predict.py # PSI drift detection unit tests
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── .env.example
```

---

## Dataset

IBM Telco Customer Churn — ~7,000 rows, 21 features, binary churn target.  
Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Place the downloaded CSV at:

```
data/external/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Raw and processed data files are excluded from version control via `.gitignore`.

---

## Engineered features

Eight business-driven features are added on top of the raw dataset in `src/features/engineer.py`.
All features are justified by domain knowledge about telecom churn behaviour.

| Feature | Type | Business rationale |
|---|---|---|
| `tenure_bucket` | Categorical | Churn risk is non-linear with tenure; new customers (≤12m) are highest risk |
| `num_services` | Numeric | More services = higher switching cost = lower churn probability |
| `has_support_services` | Binary flag | Customers with tech support or online security have lower churn rates |
| `charge_per_tenure` | Numeric | Monthly charge normalised by tenure — isolates price sensitivity |
| `monthly_spend_delta` | Numeric | Current bill vs. historical average; sudden increases signal dissatisfaction |
| `is_new_customer` | Binary flag | Tenure ≤ 6 months — highest ROI window for retention intervention |
| `contract_risk_score` | Ordinal | Month-to-month = 2, one year = 1, two year = 0 |
| `churn_binary` | Target | Yes → 1, No → 0; used as the ML training target |

The `src/features/feature_store.py` module is the single source of truth for which
columns enter the model. Phase 4 imports exclusively from there.

---

## Spark setup

The PySpark batch job connects to PostgreSQL via JDBC.  
Download the driver JAR and place it at `docker/jars/postgresql-42.7.3.jar`:

```
https://jdbc.postgresql.org/download/postgresql-42.7.3.jar
```

Ensure `JAVA_HOME` points to a Java 11+ installation before running the Spark job.

---

## Scheduling

The batch pipeline can be scheduled two ways:

**Cron (simple)** — add to your crontab with `crontab -e`:
```
0 2 * * * cd /path/to/project && .venv/bin/python -m src.pipeline.batch_predict
```

**Airflow (production)** — the DAG at `airflow/dags/churn_batch_dag.py` runs
drift detection before every scoring run and retrains automatically if PSI ≥ 0.20.
See the file header for local Airflow setup instructions.

---

## Architecture

> Full architecture diagram will be added in Phase 7.

---

## Results

Model metrics are written to `models/model_metadata.json` after training.
Full results and business impact will be documented in Phase 8.

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8469 |
| F1 (churn class) | 0.6227 |
| Recall (churn class) | 0.7834 |
