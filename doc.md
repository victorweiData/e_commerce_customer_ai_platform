E-Commerce Customer AI Platform – Project Journal

An evolving log of what was built, why, and how. Use it as a living notebook for reflection and onboarding.

⸻

1. Project Kick-off

Date	Item	Notes
2025-MM-DD	Idea & scope defined	Four business questions: late-delivery, churn, CLTV, segmentation
2025-MM-DD	Repo scaffolded	cookiecutter-data-science-style layout


⸻

2. Data Engineering

2.1 Ingestion
	•	Flow flows/ingest_olist.py uses Prefect 2 to
	1.	download the official Kaggle Olist zip via CLI;
	2.	extract into data/raw.

2.2 Cleaning
	•	Notebook + Flow flows/clean_olist.py converts raw CSVs to cleaned Parquet.
	•	Key ops
	•	Date-parsing, snake_case cols, dtypes
	•	Null handling & basic sanity tests (Great Expectations)

2.3 Warehouse load
	•	Flow flows/load_cleaned_to_db.py bulk-loads cleaned Parquet → Postgres.
	•	Credentials pulled from .env.

2.4 Feature engineering
	•	Module customer_ai/features.py exposes builders:
	•	build_sales_fact()
	•	build_order_features()
	•	build_churn_features()
	•	build_segment_features_extended()
	•	build_cltv_summary()
	•	Flow flows/feature_builders.py orchestrates the above; callable via make features.

Outcome: data/processed/*.parquet ready for modeling.

⸻

3. Infrastructure

Stack piece	Status	Reference
Docker-compose	✅ running	infra/docker-compose.yml (Postgres, Prefect, MLflow, Grafana)
Conda env	✅	environment.yml (Python 3.11)
CI/CD	✅	GitHub Actions: lint, type-check, pytest
Pre-commit	✅	black + ruff + mypy
Nightly jobs	⏳ planned	Prefect deployment & cron action


⸻

4. Modeling Roadmap

Objective	Data source	Candidate models	Metric target
Late delivery	order_features.parquet	LightGBM, CatBoost	AUC ≥ 0.80
Churn	churn_features.parquet	XGBoost, LightGBM	AUC ≥ 0.75
CLTV regression	cltv_summary.parquet	Lifetimes BG/NBD + GG	R² ≥ 0.45
Segmentation	segment_features_extended.parquet	K-Means, HDBSCAN	Silhouette ≥ 0.25


⸻

5. Lessons & Gotchas
	•	.env discipline: keep secrets out of code & commits.
	•	Peeking risk: EDA only on training split.
	•	Large files: use Git LFS for >50 MB, or exclude via .gitignore.

⸻

6. Next Steps
	1.	Parameter-sweep LightGBM for late-delivery.
	2.	Add Prefect deployment for nightly feature refresh.
	3.	Wire Evidently dashboards into Grafana.
	4.	Publish FastAPI serving stub.

⸻

This journal is auto-generated at first; feel free to append daily logs below.

⸻

Daily Logs

Date	Summary
2025-MM-DD	Initial commit, data ingested & cleaned.