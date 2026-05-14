# Telecom Customer Churn Prediction Platform

End-to-end ML system for predicting telecom customer churn — covering
data engineering, batch + real-time scoring, REST API deployment,
and a business intelligence dashboard. Built to simulate a production
environment.

---

## Tech stack

Python · PySpark · Apache Kafka · PostgreSQL · Flask · Docker · Power BI · SHAP · Airflow

---

## Quick start

**Prerequisites:** Python 3.10+, Docker Desktop, Java 11+ (for local Spark jobs)

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

# Download dataset from Kaggle and place at:
# data/external/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## Running the full pipeline

All scripts must be run as modules from the project root with the venv active.

### 1. Infrastructure

```bash
# Start Postgres, Zookeeper, Kafka, and pre-create Kafka topics
docker-compose up -d postgres zookeeper kafka kafka-init

# Confirm topics exist before continuing
docker logs churn_kafka_init
# Should list: customer-events  churn-alerts
```

### 2. ETL pipeline

```bash
# Extract → clean → transform → load into PostgreSQL
python -m src.pipeline.etl

# Verify data landed
docker exec -it churn_postgres psql -U airflow -d airflow -c \
  "SELECT COUNT(*) FROM customers;"
```

### 3. PySpark batch ETL (optional — requires Java 11+ and JAVA_HOME set)

```bash
# Download JDBC driver first:
# https://jdbc.postgresql.org/download/postgresql-42.7.3.jar
# Place at: docker/jars/postgresql-42.7.3.jar

python -m src.pipeline.spark_batch
```

### 4. EDA and feature engineering

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 5. Model training

```bash
# Compares 4 models, tunes the winner with GridSearchCV
# Expect 3–8 minutes
python -m src.models.train
```

### 6. SHAP explainability

```bash
# Requires trained model
python -m src.models.shap_analysis
```

### 7. Batch prediction

```bash
python -m src.pipeline.batch_predict

# Verify predictions
docker exec -it churn_postgres psql -U airflow -d airflow -c \
  "SELECT risk_tier, COUNT(*) FROM churn_predictions GROUP BY risk_tier;"
```

### 8. Real-time streaming pipeline

```bash
# Start Spark consumer (Linux Docker container — avoids Windows Hadoop issue)
docker-compose up -d spark-consumer

# Watch Spark logs — wait for "Streaming query started"
docker logs -f churn_spark_consumer

# Terminal A — producer (runs on host)
python -m src.streaming.producer

# Terminal B — alert consumer (start only after seeing Batch 0 in Spark logs)
python -m src.streaming.alert_consumer

# Verify alerts
docker exec -it churn_postgres psql -U airflow -d airflow -c \
  "SELECT event_type, risk_tier, COUNT(*) FROM churn_alerts GROUP BY event_type, risk_tier;"
```

### 9. Tests

```bash
pytest
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
| 6 | Real-time streaming (Kafka) | ✅ Done |
| 7 | API, dashboard & deployment | ⏳ Next |
| 8 | Documentation & packaging | ⏳ Planned |

---

## Repository layout

```text
telecom-churn-platform/
├── api/                        # Flask REST API (Phase 7)
├── dashboard/                  # Power BI dashboards (Phase 7)
├── data/
│   ├── external/               # IBM Telco Churn dataset (not committed)
│   ├── processed/              # cleaned datasets (not committed)
│   └── raw/                    # original CSVs (not committed)
├── docker/
│   ├── Dockerfile.airflow      # Airflow container
│   ├── Dockerfile.spark        # Spark consumer container (Linux, Java 21)
│   ├── init.sql                # DB schema — customers, churn_predictions, churn_alerts
│   └── jars/                   # PostgreSQL JDBC driver (not committed)
├── docs/
│   ├── architecture/           # System architecture diagrams (Phase 7)
│   └── screenshots/            # EDA, model evaluation, SHAP plots
├── models/                     # Trained model artifacts (not committed)
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory data analysis and feature engineering
├── airflow/
│   └── dags/
│       └── churn_batch_dag.py  # Daily batch scoring + drift-gated retraining DAG
├── src/
│   ├── features/
│   │   ├── engineer.py         # 8 business-driven engineered features
│   │   └── feature_store.py    # Single source of truth for feature column lists
│   ├── models/
│   │   ├── preprocessor.py     # sklearn ColumnTransformer pipeline
│   │   ├── train.py            # 4-model comparison, GridSearchCV, saves model + metadata
│   │   ├── evaluate.py         # Confusion matrix, ROC, precision-recall plots
│   │   ├── predict.py          # predict_batch() and predict_single() — used by API + streaming
│   │   └── shap_analysis.py    # SHAP beeswarm + waterfall explainability plots
│   ├── pipeline/
│   │   ├── etl.py              # Pipeline orchestrator
│   │   ├── extract.py          # CSV ingestion
│   │   ├── transform.py        # Cleaning and standardisation
│   │   ├── load.py             # PostgreSQL loader — contains COLUMN_MAP
│   │   ├── spark_batch.py      # PySpark alternative ETL via JDBC
│   │   ├── batch_predict.py    # Batch scoring — reads DB, writes churn_predictions
│   │   └── drift_detector.py   # PSI-based feature drift detection
│   ├── streaming/
│   │   ├── topics.py           # Kafka topic constants and event type list
│   │   ├── producer.py         # CRM event simulator — publishes to customer-events
│   │   ├── spark_consumer.py   # Spark Structured Streaming — scores micro-batches
│   │   └── alert_consumer.py   # Reads churn-alerts topic, writes to churn_alerts table
│   └── utils/
│       ├── config.py           # All env vars — DB, Kafka, file paths
│       └── logger.py           # Loguru structured logging
├── tests/
│   ├── test_pipeline.py        # ETL transform unit tests
│   ├── test_features.py        # Feature engineering unit tests
│   ├── test_model.py           # Model inference and risk tier tests
│   ├── test_batch_predict.py   # PSI drift detection tests
│   └── test_streaming.py       # Streaming event and alert record tests
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
├── requirements-airflow.txt
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

Eight business-driven features added in `src/features/engineer.py`.
All are justified by domain knowledge about telecom churn behaviour.

| Feature | Type | Business rationale |
|---|---|---|
| `tenure_bucket` | Categorical | Churn risk is non-linear — new customers (≤12m) churn at ~48% |
| `num_services` | Numeric | More services = higher switching cost = lower churn probability |
| `has_support_services` | Binary | Customers with tech support or online security churn measurably less |
| `charge_per_tenure` | Numeric | Monthly charge normalised by tenure — isolates price sensitivity |
| `monthly_spend_delta` | Numeric | Current bill vs historical average — sudden increases signal dissatisfaction |
| `is_new_customer` | Binary | Tenure ≤ 6 months — highest ROI window for retention intervention |
| `contract_risk_score` | Ordinal | Month-to-month = 2, one year = 1, two year = 0 |
| `churn_binary` | Target | Yes → 1, No → 0; training target only |

`src/features/feature_store.py` is the single source of truth for which columns enter the model.

---

## Model results

Four candidate models compared via 5-fold stratified cross-validation.
Winner selected by ROC-AUC and tuned with GridSearchCV.
Metrics written to `models/model_metadata.json` after training.

| Metric | Value |
|---|---|
| ROC-AUC | 0.8469 |
| F1 (churn class) | 0.6227 |
| Recall (churn class) | 0.7834 |

**Why recall is the primary metric:**
A false negative (missed churner) costs ~£780 ARR. A false positive (unnecessary retention call) costs ~£20.
The 39:1 cost ratio justifies optimising for recall over accuracy.

---

## Streaming pipeline

The real-time pipeline consists of three components:

| Component | File | Role |
|---|---|---|
| Producer | `src/streaming/producer.py` | Reads customers from PostgreSQL, attaches a CRM event type, publishes to `customer-events` at 0.5s intervals |
| Spark consumer | `src/streaming/spark_consumer.py` | Reads `customer-events` in 10s micro-batches, scores via `predict_batch()`, publishes high-risk alerts to `churn-alerts` |
| Alert consumer | `src/streaming/alert_consumer.py` | Reads `churn-alerts`, writes each alert to the `churn_alerts` PostgreSQL table |

**Network rule:** Services inside Docker use `kafka:9092`. Host processes use `localhost:29092`. Never mix these.

**Run the Spark consumer via Docker only.** Running it locally on Windows crashes with `NativeIO$Windows.access0` — a Hadoop native binary issue that doesn't exist in the Linux container.

```bash
# Correct
docker-compose up -d spark-consumer

# Never run this on Windows
python -m src.streaming.spark_consumer
```

---

## Docker services

| Service | Container | Port | Purpose |
|---|---|---|---|
| postgres | churn_postgres | 5432 | Shared database |
| zookeeper | churn_zookeeper | 2181 | Kafka coordination |
| kafka | churn_kafka | 29092 (host) / 9092 (internal) | Message broker |
| kafka-init | churn_kafka_init | — | Pre-creates topics on startup |
| spark-consumer | churn_spark_consumer | — | Streaming inference (Linux) |
| airflow-webserver | airflow-webserver | 8080 | Airflow UI |
| airflow-scheduler | airflow-scheduler | — | DAG execution |

---

## PostgreSQL tables

| Table | Populated by | Contains |
|---|---|---|
| `customers` | `etl.py` | Clean customer records from IBM dataset |
| `churn_predictions` | `batch_predict.py` | Batch scored customers with churn_prob and risk_tier |
| `churn_alerts` | `alert_consumer.py` | Real-time high-risk alerts from streaming pipeline |

---

## Scheduling

**Cron (simple):**
```
0 2 * * * cd /path/to/project && .venv/bin/python -m src.pipeline.batch_predict
```

**Airflow (production):** The DAG at `airflow/dags/churn_batch_dag.py` runs daily at 02:00 UTC.
It checks PSI drift before every scoring run and retrains automatically if PSI ≥ 0.20.

---

## Spark setup (local batch job only)

The PySpark batch job (`spark_batch.py`) connects to PostgreSQL via JDBC.
Download the driver JAR and place it at `docker/jars/postgresql-42.7.3.jar`:

```
https://jdbc.postgresql.org/download/postgresql-42.7.3.jar
```

Ensure `JAVA_HOME` points to a Java 11+ installation before running.
The streaming Spark consumer runs inside Docker and does not require local Java.

---

## Architecture

> Full architecture diagram will be added in Phase 7.