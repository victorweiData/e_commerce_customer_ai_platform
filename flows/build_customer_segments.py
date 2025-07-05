"""
Prefect flow: build 5-segment customer clusters
------------------------------------------------
Run once:
    python -m flows.build_customer_segments
Or via Make:
    make build-segments
"""

from __future__ import annotations
from datetime import timedelta
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

load_dotenv()

# ── paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
PROC_DIR      = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models" / "segmentation"
REPORTS_DIR   = PROJECT_ROOT / "reports" / "segmentation"
FIGURES_DIR   = REPORTS_DIR / "figures"
SEG_FILE      = PROC_DIR / "customer_segments.parquet"
SUMMARY_FILE  = PROC_DIR / "segment_summary.json"

# where to drop into your React app so Vite will serve it
FRONT_PUBLIC  = PROJECT_ROOT / "dashboard" / "public" / "segment_summary.json"

# ── feature list (keep in one place) ───────────────────────────────────
BEHAV_FEATURES = [
    "frequency",
    "avg_order_value",
    "tenure_days",
    "avg_installments",
    "payment_diversity",
    "avg_review_score",
    "review_rate",
    "category_diversity",
]

# ── tasks ──────────────────────────────────────────────────────────────
@task(
    name="load_customer_master",
    retries=2,
    retry_delay_seconds=5,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
)
def load_master() -> pd.DataFrame:
    return pd.read_parquet(PROC_DIR / "customer_master.parquet")


@task(name="fit_kmeans")
def fit_kmeans(df: pd.DataFrame, k: int = 5) -> pd.Series:
    X = df[BEHAV_FEATURES].fillna(0).to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=k, n_init=50, random_state=42)
    labels = km.fit_predict(X_scaled)

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, MODELS_DIR / f"kmeans_k{k}.pkl")

    return pd.Series(labels, index=df.index, name="cluster")


@task(name="save_outputs")
def save_outputs(
    df_master: pd.DataFrame,
    clusters: pd.Series,
) -> None:
    log = get_run_logger()

    # 1️⃣ customer-level parquet
    seg_df = df_master["customer_id"].to_frame()
    seg_df["cluster"] = clusters.values
    seg_df.to_parquet(SEG_FILE, index=False)
    log.info("💾 Saved %s", SEG_FILE.name)

    # 2️⃣ summary JSON for frontend & data/processed
    summary = (
        df_master.assign(cluster=clusters)
        .groupby("cluster")
        .agg(
            customers=("customer_id", "count"),
            percentage=("customer_id", lambda s: round(100 * len(s) / len(df_master), 1)),
            avgOrderValue=("avg_order_value", "mean"),
            avgInstallments=("avg_installments", "mean"),
            avgReviewScore=("avg_review_score", "mean"),
            reviewRate=("review_rate", "mean"),
            categoryDiversity=("category_diversity", "mean"),
        )
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    payload = json.dumps(summary, indent=2)

    # write to data/processed
    SUMMARY_FILE.write_text(payload)
    log.info("💾 Saved %s", SUMMARY_FILE.name)

    # also write to your React public folder (used for dashboard)
    FRONT_PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    FRONT_PUBLIC.write_text(payload)
    log.info("💾 Saved %s for React", FRONT_PUBLIC.name)

    # 3️⃣ recommendation & figures to reports/segmentation
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "recommendations.json").write_text(payload)
    log.info("💾 Saved segmentation summary as recommendation")


# ── flow ───────────────────────────────────────────────────────────────
@flow(name="build_customer_segments_k5")
def build_customer_segments() -> None:
    logger = get_run_logger()
    master   = load_master()
    clusters = fit_kmeans(master, k=5)
    save_outputs(master, clusters)
    logger.info("✅ Built customer segments (k=5)")


if __name__ == "__main__":
    build_customer_segments()
