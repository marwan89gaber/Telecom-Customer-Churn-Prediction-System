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

# Start PostgreSQL
docker-compose up -d

# Download dataset from Kaggle and place at:
# data/external/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## Running the pipeline

All scripts must be run as modules from the project root with the venv active.

```bash
# Run the full ETL pipeline (extract → transform → load → PostgreSQL)
python -m src.pipeline.etl

# Verify data landed in the database
docker exec -it churn_postgres psql -U churn_admin -d churn_db \
  -c "SELECT churn, COUNT(*) FROM customers GROUP BY churn;"

# Run the PySpark batch job (requires Java 11+ and JAVA_HOME set)
python -m src.pipeline.spark_batch

# Run tests
pytest

# Run the EDA notebook
jupyter notebook notebooks/01_eda.ipynb
```

---

## Project phases

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Foundation & setup | ✅ Done |
| 2 | Data engineering pipeline | ✅ Done |
| 3 | EDA & feature engineering | ✅ Done |
| 4 | ML modelling & evaluation | ⏳ Planned |
| 5 | Batch prediction pipeline | ⏳ Planned |
| 6 | Real-time streaming (Kafka) | ⏳ Planned |
| 7 | API, dashboard & deployment | ⏳ Planned |
| 8 | Documentation & packaging | ⏳ Planned |

---

## Repository layout

telecom-churn-platform/
├── data/
│   ├── raw/              # original CSVs (not committed)
│   ├── processed/        # cleaned outputs (not committed)
│   └── external/         # IBM Telco Churn dataset
├── src/
│   ├── pipeline/         # ETL: extract, transform, load, spark batch
│   ├── features/         # feature engineering (phase 3)
│   ├── models/           # training and evaluation (phase 4)
│   ├── streaming/        # Kafka producer/consumer (phase 6)
│   └── utils/            # config, logger
├── api/                  # Flask REST API (phase 7)
├── models/               # saved model artifacts (phase 4)
├── notebooks/            # EDA and experimentation (phase 3)
├── dashboard/            # Power BI files (phase 7)
├── docker/               # Dockerfiles, init scripts, JDBC jar
├── docs/                 # architecture diagrams and screenshots
├── tests/                # unit and integration tests
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
└── .env.example

---

## Dataset

IBM Telco Customer Churn — ~7,000 rows, 21 features, binary churn target.
Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Place the CSV at: `data/external/WA_Fn-UseC_-Telco-Customer-Churn.csv`

---

## Architecture

> Architecture diagram will be added in Phase 7.

---

## Results

> Model metrics and business impact will be documented after Phase 4.