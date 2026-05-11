# Telecom Customer Churn Prediction System

A practical, engineering-first project demonstrating a reproducible churn prediction platform built from raw data ingestion through model deployment. The guide below documents the planned phases, how to run the repository, and the project layout so reviewers and hiring managers can evaluate both data science and engineering maturity.

## Table of contents

- Project overview
- Quick start
- Phases & deliverables
- Repository layout
- Contributing
- License & contact

## Project overview

This repository implements an end-to-end churn prediction system using the IBM Telco Churn dataset (≈7k rows, 21 features). The emphasis is on engineering: reproducible ETL, batch and streaming scoring, containerized services, and clear deployment artifacts — not just notebooks.

## Architecture
<img width="2125" height="2250" alt="phase1_folder_structure" src="https://github.com/user-attachments/assets/aa5c820d-b513-4865-951b-bef48c5d9ce2" />

## Quick start

Prerequisites: Python 3.8+, Docker (optional), Java + Spark (for Spark jobs)

1. Create a virtual environment and install Python deps:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run core scripts (examples):

```bash
python main.py            # project entry or demo runner
python Social\ Scope/main.py   # example module (adjust path as needed)
```

3. Docker and orchestration: see `docker/` or `docker-compose.yml` when available (phase artifacts).

## Phases & deliverables

Each phase focuses on one set of deliverables — engineering-first and resume-ready.

1. Foundation & setup
	- Repository scaffold, `requirements.txt`, `.gitignore`, Docker skeleton, README.
2. Data engineering pipeline
	- Production-style ETL (CSV → clean → transform → PostgreSQL) and a PySpark batch job.
3. EDA & feature engineering
	- Business-focused analysis, feature creation, imbalance handling, screenshots.
4. ML modelling & evaluation
	- Train and compare multiple models, tune hyperparameters, use recall as primary metric, explainers (SHAP).
5. Batch prediction pipeline
	- Nightly Spark scoring job, risk tier writes, PSI-based retrain trigger, Airflow DAG example.
6. Real-time streaming
	- Kafka event producer, Spark Structured Streaming consumer, alert topic for high-risk users, Dockerized.
7. API, dashboard & deployment
	- Flask API (3 endpoints), Docker Compose orchestration, dashboard examples (Power BI), Swagger.
8. Documentation & packaging
	- Polished README, release tag, short Loom demo, portfolio-ready write-up.

## Repository layout

Top-level files and folders (high-level):

- `README.md` — this file
- `requirements.txt` — Python dependencies
- `main.py` — project runner / demo entry
- `config.py` — project configuration
- `Social Scope/` — core application code and modules
- `media/`, `platforms/`, `storage/`, `utils/` — helpers and integrations
- `data/`, `inputs/`, `outputs/` — data ingestion and processed outputs

Adjust paths and commands to your environment; relative paths use Windows separators in examples above.

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repo
2. Create a feature branch
3. Add tests or small runnable examples
4. Open a PR with a clear description of changes

For major changes (new services, schema updates), open an issue first to discuss design.

## License & contact

This repository is a project scaffold and learning exercise.

---
