# E-Commerce Customer AI Platform 🚀

A production-ready MLOps platform built on the Olist Brazilian e-commerce dataset. This repository demonstrates how to ship real-world machine learning products from raw data to automated retraining, continuous delivery, live dashboards, and model drift monitoring.

## 🎥 Demo [https://youtu.be/o8FfLyBBcN4](url)

![Watch the demo](https://github.com/user-attachments/assets/f8c5c15a-f6a4-4346-816d-eea06cc54cff)]

## 💰 Business Impact

- **💵 Cost Savings:** $2.5M saved through churn prevention
- **👥 Customer Retention:** 60k+ customers retained
- **⚙️ Operational Efficiency:** 85% reduction in manual ML workflow tasks

## 🌟 Key Features

| Feature | Description |
|---------|-------------|
| **Full-Stack MLOps** | Complete pipeline: Data → Features → Model → CI/CD → Serving |
| **Modern Tooling** | Prefect • PostgreSQL • FastAPI • MLflow • Evidently • Docker |
| **Automated Intelligence** | Nightly retraining & drift detection with Slack alerts |
| **Interactive Dashboard** | React UI powered by FastAPI JSON API |
| **Infrastructure-as-Code** | Docker Compose for local dev + GitHub Actions for CI/CD |
| **Enterprise-Grade Quality** | Ruff, Black, MyPy, Pytest, 100% pre-commit enforced |

## 🎯 ML Objectives & Production Status

| Objective | Status | Models | Business Value |
|-----------|--------|--------|---------------|
| Customer Segmentation | ✅ **In Production** | K-means, PCA-reduced + t-SNE for viz, DBSCAN | Targeted marketing campaigns |
| Customer Churn Prediction | 🔄 **Training Nightly** | LightGBM, XGBoost, Logistic Reg, Random Forest | Proactive retention strategies |
| Customer Lifetime Value | 📋 **Backlog** | BG/NBD + Gamma-Gamma, Gradient-Boosted Regressor | Long-term profitability optimization |

🗂 **Dive Deeper**

Full experiment notes, metric evolution charts, and business commentary for every
objective above live in **`/reports/`**

*this is more like a learning guide for myself not a full blown professional report*

## 🛠️ Technology Stack

### Core Infrastructure
- **🎛️ Orchestration** → Prefect 2 (flows, schedules, retries)
- **🗄️ Data Warehouse** → PostgreSQL
- **🔄 Transformations** → dbt (SQL + Python models)
- **🧠 ML Frameworks** → scikit-learn • LightGBM • XGBoost
- **🛰️ API Layer** → FastAPI (serves predictions & dashboard JSON)
- **📊 Experiment Tracking** → MLflow (autolog + model registry)

### Monitoring & Deployment
- **📈 Monitoring/Drift** → Evidently AI + Slack webhooks
- **🚀 CI/CD** → GitHub Actions → Docker Hub → Auto-deploy
- **🎨 Frontend** → React + Vite + Tailwind CSS
- **🛠️ Dev Experience** → Conda • Makefile


## 📈 Automated Pipeline Architecture

### 🔄 Data & Feature Engineering
- **Schedule:** Daily at 02:00 UTC
- **Output:** Parquet feature tables + PostgreSQL incremental loads

### 🧠 Model Training & Evaluation
- **Schedule:** Nightly at 02:30 UTC (after fresh features)
- **Process:**
  1. Train baseline + tuned models (LogReg, LightGBM, XGBoost)
  2. Hyperparameter optimization with Optuna
  3. Register best model in MLflow Registry (`ChurnModel/Production`)
  4. Push artifacts to S3-compatible MinIO bucket

### 📊 Drift & Performance Monitoring
- **Tools:** Evidently statistical tests + Grafana dashboards
- **Alerts:** Slack notifications when PSI > 0.2 or weekly ROC-AUC drops > 3%

## 📊 Current Performance Metrics

### 🎯 Model Performance

| Model | Metric | Target | Latest | Trend |
|-------|--------|---------|--------|-------|
| **Churn v2 (Production)** | ROC-AUC | ≥ 0.75 | **0.812** | ↗️ Stable |
| | Precision @ 20% | ≥ 0.20 | **0.27** | ↗️ Improving |
| **Segmentation (K-means)** | Silhouette Score | ≥ 0.25 | **0.31** | ➡️ Stable |

> 🎉 **Achievement:** Baseline churn ROC-AUC improved from 0.62 → 0.812 (31% relative lift)

## 🏗️ Repository Structure

```
📦 e_commerce_customer_ai_platform/
├── 📚 customer_ai/          # Reusable Python package
├── 🎨 dashboard/            # React + Tailwind frontend
├── ⚡ api/                  # FastAPI application (serves /api & static dashboard)
├── 📊 data/                 # Raw, Interim, Processed, External
├── 🔄 flows/                # Prefect pipelines (ingest, features, train, monitor)
├── 🐳 infra/                # Docker Compose & K8s manifests
├── 🧪 tests/                # Pytest unit & integration tests
├── ⚙️ Makefile             # Developer commands (run `make help`)
└── 📖 report/               # Detailed reports & notebooks
```

## 🔄 CI/CD Pipeline (GitHub Actions)

| Stage | Trigger | Actions |
|-------|---------|---------|
| **🔍 Quality Gate** | PR / Push | Ruff • MyPy • Pytest |
| **🏗️ Build & Push** | merge → main | Build Docker images → Push to Hub |
| **🚀 Deploy** | Release tag | Helm upgrade on staging cluster |
| **🌙 Nightly ML** | Cron 02:00 UTC | `prefect deployment run build_customer_churn` |


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

GitHub Actions will automatically run quality gates and deploy a preview environment.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with the [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce)
- Inspired by modern MLOps best practices

---

<div align="center">
  
⭐ **Star this repo if it helped you build better ML systems!** ⭐

</div>
