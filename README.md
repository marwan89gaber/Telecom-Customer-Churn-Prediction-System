# Telecom Customer Churn Prediction Platform

End-to-end ML system for predicting telecom customer churn — covering
data engineering, batch + real-time scoring, REST API deployment,
and a business intelligence dashboard. Built to simulate a Vodafone
production environment.

---

## Tech stack

Python · PySpark · Apache Kafka · PostgreSQL · Flask · Docker · Power BI · SHAP

---

## Quick start

Prerequisites: Python 3.10+, Docker Desktop

```bash
git clone https://github.com/yourname/telecom-churn-platform
cd telecom-churn-platform

# Environment
cp .env.example .env

# Start PostgreSQL
docker-compose up -d

# Python dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Project phases

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Foundation & setup | ✅ Done |
| 2 | Data engineering pipeline | 🔄 In progress |
| 3 | EDA & feature engineering | ⏳ Planned |
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
│   ├── pipeline/         # ETL scripts
│   ├── features/         # feature engineering
│   ├── models/           # training and evaluation
│   ├── streaming/        # Kafka producer/consumer
│   └── utils/            # shared helpers
├── api/                  # Flask REST API
├── models/               # saved model artifacts
├── notebooks/            # EDA and experimentation
├── dashboard/            # Power BI files
├── docker/               # Dockerfiles and init scripts
├── docs/                 # architecture diagrams and screenshots
├── tests/                # unit and integration tests
├── docker-compose.yml
├── requirements.txt
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
