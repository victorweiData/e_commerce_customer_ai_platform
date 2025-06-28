E-Commerce Customer AI Platform

End-to-end MLOps demo built on the public Olist Brazilian E-commerce dataset. It walks through the whole lifecycle—from raw data ingestion to model serving & monitoring—using production-style tooling.

⸻

📁 Repository layout

├── customer_ai/            ← Python package (config, features, models)
│   ├── data/               ← Feature helpers & dataset joins
│   └── modeling/           ← `train.py` / `predict.py`
├── data/                   ← raw / interim / processed parquet
├── flows/                  ← Prefect flows (ingest, clean, train, monitor)
├── infra/                  ← docker-compose stack (Postgres, Prefect, MLflow, Grafana)
├── docs/                   ← MkDocs site (optional)
├── tests/                  ← pytest unit tests
├── environment.yml         ← Conda environment spec
├── Makefile                ← one-liners: lint, test, ingest, db-load …
└── .github/workflows/ci.yml← pre-commit & pytest


⸻

🚀 Quick-start (local)

# 1️⃣  Clone & create environment
conda env create -f environment.yml -n e_comm_ai
conda activate e_comm_ai

# 2️⃣  Bring up infra stack (detached)
docker compose -f infra/docker-compose.yml up -d

# 3️⃣  Download + unzip raw Olist data
python -m flows.ingest_olist          # or: make ingest-fast

# 4️⃣  Clean raw → processed parquet
make clean-data                       # runs Prefect `flows.clean_olist`

# 5️⃣  Load processed parquet → Postgres warehouse
make db-load                          # runs Prefect `flows.load_cleaned_to_db`

# 6️⃣  Train baseline models & record to MLflow
python -m flows.train_models          # WIP

# 7️⃣  Lint, type-check, test
make format lint test

Service	URL	Default creds
MLflow Tracking	http://localhost:5000	–
Prefect Orion UI	http://localhost:4200	–
Grafana	http://localhost:3000	admin / admin
Postgres	localhost:5432/olist	see .env


⸻

🔑 Environment variables (.env)

Create a simple text file named .env in the project root (sibling to Makefile). This file is ignored by Git, so your credentials stay local.

PG_HOST=localhost
PG_PORT=5432
PG_USER=olist
PG_PASSWORD=olist
PG_DB=olist

These variables are automatically picked up by Prefect flows, Docker Compose, and your notebooks.

⸻

🔄 CI/CD
	•	GitHub Actions (ci.yml) installs the Conda environment, then runs pre-commit (black + ruff + mypy) and pytest on every push/PR to main.
	•	Future: add a deployment job that triggers a Prefect deployment or pushes a Docker image to a registry.

⸻

📊 Business objectives & baseline models

#	Objective	Model family	Performance gate
1	Late-delivery prediction	LightGBM classifier	ROC-AUC ≥ 0.80
2	Customer churn risk	XGBoost classifier	ROC-AUC ≥ 0.75
3	Customer LTV (CLTV)	BG/NBD + GammaGamma	R² ≥ 0.45
4	Customer segmentation	K-Means / HDBSCAN	Silhouette ≥ 0.25


⸻

🛠️ Tech stack
	•	Prefect 2 – orchestration
	•	dbt – SQL transformations & tests
	•	Postgres & DuckDB – warehouse/local analytics
	•	MLflow – experiment tracking & model registry
	•	LightGBM / XGBoost / Lifetimes – modeling libs
	•	FastAPI – realtime serving (roadmap)
	•	Evidently + Grafana – data & model drift dashboards
	•	GitHub Actions – CI gates

⸻

🌐 References
	•	Olist dataset – https://www.kaggle.com/olistbr/brazilian-ecommerce
	•	Prefect docs – https://docs.prefect.io
	•	dbt docs – https://docs.getdbt.com

⸻

Questions or ideas? Feel free to open an issue or PR!