# E-Commerce Customer AI Platform

An end-to-end MLOps demo built on the public Olist Brazilian E-commerce dataset. This project demonstrates the complete machine learning lifecycle—from raw data ingestion to model serving and monitoring—using production-grade tooling and best practices.

## 🎯 Overview

This platform implements four key customer analytics models:
- **Late delivery prediction** - Identify orders at risk of late delivery
- **Customer churn risk** - Predict which customers are likely to churn  
- **Customer lifetime value (CLTV)** - Estimate long-term customer value
- **Customer segmentation** - Group customers for targeted marketing

## 📁 Repository Structure

```
├── customer_ai/              # Core Python package
│   ├── data/                 # Feature engineering & dataset utilities
│   └── modeling/             # Model training & prediction modules
├── data/                     # Data storage (raw/interim/processed parquet files)
├── flows/                    # Prefect orchestration workflows
├── infra/                    # Infrastructure (Docker Compose stack)
├── docs/                     # Documentation (MkDocs)
├── tests/                    # Unit tests (pytest)
├── license/                  # License information
├── environment.yml           # Conda environment specification
├── Makefile                  # Development shortcuts
└── .github/workflows/        # CI/CD pipelines
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Conda or Miniconda
- Git

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd e-commerce-customer-ai

# Create and activate conda environment
conda env create -f environment.yml -n e_comm_ai
conda activate e_comm_ai
```

### 2. Infrastructure Setup
```bash
# Start the infrastructure stack (Postgres, Prefect, MLflow, Grafana)
docker compose -f infra/docker-compose.yml up -d
```

### 3. Data Pipeline
```bash
# Download and extract Olist dataset
python -m flows.ingest_olist
# Alternative: make ingest-fast

# Clean and process raw data
make clean-data

# Load processed data to Postgres warehouse
make db-load
```

### 4. Model Training
```bash
# Train baseline models and log to MLflow
python -m flows.train_models
```

### 5. Development Tools
```bash
# Code formatting, linting, and testing
make format lint test
```

## 🌐 Service Access

| Service | URL | Credentials |
|---------|-----|-------------|
| MLflow Tracking | http://localhost:5000 | None required |
| Prefect UI | http://localhost:4200 | None required |
| Grafana | http://localhost:3000 | admin / admin |
| Postgres | localhost:5432/olist | See .env configuration |

## ⚙️ Configuration

Create a `.env` file in the project root with the following variables:

```bash
PG_HOST=localhost
PG_PORT=5432
PG_USER=olist
PG_PASSWORD=olist
PG_DB=olist
```

> **Note:** The `.env` file is git-ignored to keep credentials secure.

## 📊 Model Performance Targets

| Objective | Model Type | Performance Gate |
|-----------|------------|------------------|
| Late delivery prediction | LightGBM Classifier | ROC-AUC ≥ 0.80 |
| Customer churn risk | XGBoost Classifier | ROC-AUC ≥ 0.75 |
| Customer LTV (CLTV) | BG/NBD + Gamma-Gamma | R² ≥ 0.45 |
| Customer segmentation | K-Means / HDBSCAN | Silhouette ≥ 0.25 |

## 🛠️ Technology Stack

### Core Infrastructure
- **Orchestration:** Prefect 2
- **Data Warehouse:** Postgres & DuckDB
- **Experiment Tracking:** MLflow
- **Monitoring:** Evidently + Grafana

### Data & ML
- **Transformations:** dbt
- **ML Libraries:** LightGBM, XGBoost, Lifetimes
- **Serving:** FastAPI (roadmap)

### Development
- **CI/CD:** GitHub Actions
- **Testing:** pytest
- **Code Quality:** black, ruff, mypy

## 🔄 CI/CD Pipeline

The GitHub Actions workflow (`ci.yml`) automatically:
- Sets up the Conda environment
- Runs code quality checks (black, ruff, mypy)
- Executes the test suite with pytest
- Triggers on every push/PR to main branch

### Roadmap
- [ ] Deployment automation for Prefect flows
- [ ] Docker image builds and registry pushes
- [ ] Model performance monitoring alerts

## 📚 References

- [Olist Brazilian E-commerce Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce)
- [Prefect Documentation](https://docs.prefect.io)
- [dbt Documentation](https://docs.getdbt.com)

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Open issues for bugs or feature requests
- Submit pull requests for improvements
- Share feedback and suggestions

## 📄 License

See the [LICENSE](LICENSE) for license information.

---

**Questions or ideas?** Open an issue or submit a PR!