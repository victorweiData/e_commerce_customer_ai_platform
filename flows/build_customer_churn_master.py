#!/usr/bin/env python
# flows/build_customer_churn_master.py
"""Prefect flow that builds **train/val/test** churn-feature tables
and writes them to `data/processed/`.

▪️ Produces three parquet files:
      • `churn_train_seed42.parquet`
      • `churn_val_seed42.parquet`
      • `churn_test_seed42.parquet`
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
from dotenv import load_dotenv
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from sklearn.model_selection import train_test_split

# ── project imports ──────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parents[1]))  # project root
from customer_ai.config import PROCESSED_DATA_DIR  # noqa: E402

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEED          = 42
CHURN_MONTHS  = 6
TEST_SIZE     = 0.10
VAL_SIZE      = 0.10  # fraction of total

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def delivery_features(orders: pd.DataFrame) -> pd.DataFrame:
    df = (
        orders.query("order_status=='delivered'")
              .dropna(subset=["order_delivered_customer_date","order_estimated_delivery_date"])
              .copy()
    )
    
    if df.empty:
        return pd.DataFrame()
    
    df["days_diff"] = (df.order_delivered_customer_date - df.order_estimated_delivery_date).dt.days
    df["actual_days"] = (df.order_delivered_customer_date - df.order_purchase_timestamp).dt.days

    # Focus on both promised delivery performance AND absolute delivery speed
    g = df.groupby("customer_id").agg({
        "days_diff": ["mean", "std"],     # Average delay and consistency
        "actual_days": ["mean", "std"],   # Average delivery time and consistency
        "order_id": "count",              # Number of delivered orders
    }).round(2)
    
    # Flatten column names
    g.columns = [f"delivery_{c[0]}_{c[1]}" for c in g.columns]
    
    # Late delivery metrics (relative to promise)
    late_orders = df[df.days_diff > 0].groupby("customer_id").size()
    g["delivery_late_ratio"] = (late_orders / g.delivery_order_id_count).fillna(0)
    g["delivery_consistently_late"] = (g.delivery_late_ratio > 0.5).astype(int)
    
    # NEW: Absolute delivery speed metrics (regardless of promise)
    # These capture customer frustration with slow delivery even if "on time"
    
    # Flag customers who experienced slow deliveries (>7 days is often considered slow for e-commerce)
    slow_orders = df[df.actual_days > 7].groupby("customer_id").size()
    g["delivery_slow_ratio"] = (slow_orders / g.delivery_order_id_count).fillna(0)
    g["delivery_frequently_slow"] = (g.delivery_slow_ratio > 0.3).astype(int)
    
    # Flag customers who experienced very slow deliveries (>14 days)
    very_slow_orders = df[df.actual_days > 14].groupby("customer_id").size()
    g["delivery_very_slow_ratio"] = (very_slow_orders / g.delivery_order_id_count).fillna(0)
    g["delivery_has_very_slow"] = (g.delivery_very_slow_ratio > 0).astype(int)
    
    # Average delivery speed categories - create binary flags directly
    g["delivery_speed_fast"] = (g.delivery_actual_days_mean <= 3).astype(int)
    g["delivery_speed_normal"] = ((g.delivery_actual_days_mean > 3) & (g.delivery_actual_days_mean <= 7)).astype(int)
    g["delivery_speed_slow"] = ((g.delivery_actual_days_mean > 7) & (g.delivery_actual_days_mean <= 14)).astype(int)
    g["delivery_speed_very_slow"] = (g.delivery_actual_days_mean > 14).astype(int)
    
    # Delivery consistency (high std means unpredictable delivery times)
    g["delivery_inconsistent"] = (g.delivery_actual_days_std > 5).astype(int)
    
    # Combined dissatisfaction score (accounts for both late and slow deliveries)
    g["delivery_dissatisfaction_score"] = (
        g.delivery_late_ratio * 0.3 +           # Being late vs promise
        g.delivery_slow_ratio * 0.4 +           # Being slow in absolute terms  
        g.delivery_very_slow_ratio * 0.3        # Having very slow orders
    )
    
    g["delivery_likely_dissatisfied"] = (g.delivery_dissatisfaction_score > 0.3).astype(int)
    
    # Clean up - keep most important columns
    final_cols = [
        "delivery_days_diff_mean",        # Average delay vs promise
        "delivery_actual_days_mean",      # Average actual delivery time
        "delivery_actual_days_std",       # Delivery time consistency
        "delivery_late_ratio",            # Ratio of late orders
        "delivery_slow_ratio",            # Ratio of slow orders (>7 days)
        "delivery_very_slow_ratio",       # Ratio of very slow orders (>14 days)
        "delivery_consistently_late",     # Binary: often late vs promise
        "delivery_frequently_slow",       # Binary: often slow in absolute terms
        "delivery_has_very_slow",         # Binary: has very slow orders
        "delivery_inconsistent",          # Binary: inconsistent delivery times
        "delivery_dissatisfaction_score", # Combined dissatisfaction metric
        "delivery_likely_dissatisfied",   # Binary: likely dissatisfied overall
        "delivery_speed_fast",            # Binary: average delivery <= 3 days
        "delivery_speed_normal",          # Binary: average delivery 3-7 days
        "delivery_speed_slow",            # Binary: average delivery 7-14 days
        "delivery_speed_very_slow",       # Binary: average delivery > 14 days
    ]
    
    return g[final_cols].fillna(0)


def review_features(reviews: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    if reviews.empty:
        return pd.DataFrame()
    
    # Only process reviews with valid scores (most important signal)
    rv = reviews.dropna(subset=["review_score"]).copy()
    
    if rv.empty:
        return pd.DataFrame()
    
    # Add customer_id
    rv = rv.merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
    
    # Focus on 3 core review metrics
    g = rv.groupby("customer_id").agg({
        "review_score": ["mean", "min"],  # Average satisfaction and worst experience
        "order_id": "count",              # Number of reviews
    }).round(2)
    
    g.columns = [f"review_{c[0]}_{c[1]}" for c in g.columns]
    
    # Simple binary flags for dissatisfaction
    bad_reviews = rv[rv.review_score <= 2].groupby("customer_id").size()
    g["review_bad_ratio"] = (bad_reviews / g.review_order_id_count).fillna(0)
    g["review_dissatisfied"] = (g.review_review_score_mean < 3.5).astype(int)
    
    return g.drop(columns=["review_order_id_count"])


def rfm_features_historical_window(orders: pd.DataFrame,
                                  items: pd.DataFrame,
                                  payments: pd.DataFrame,
                                  is_train: bool,
                                  thresholds: dict = None,
                                  window_months: int = 6):
    """
    Alternative approach: Use a specific historical window before the churn prediction point.
    
    This creates a gap between the features and the churn definition period.
    """
    if orders.empty:
        return pd.DataFrame(), thresholds
    
    # Use data from 6+ months ago (before the churn definition period)
    latest = orders.order_purchase_timestamp.max()
    window_end = latest - pd.DateOffset(months=window_months)
    window_start = window_end - pd.DateOffset(months=window_months)
    
    # Only use data from this historical window
    window_orders = orders[
        (orders.order_purchase_timestamp >= window_start) & 
        (orders.order_purchase_timestamp <= window_end)
    ].copy()
    
    if window_orders.empty:
        return pd.DataFrame(), thresholds
    
    # Build features on this historical window
    od = window_orders.merge(items, on="order_id", how="left")
    pay = payments.groupby("order_id")["payment_value"].sum().reset_index()
    od = od.merge(pay, on="order_id", how="left")

    # Historical behavior patterns
    g = od.groupby("customer_id").agg({
        "order_purchase_timestamp": "count",
        "payment_value": ["sum", "mean"],
        "freight_value": "sum",
        "price": "sum",
    }).round(2)
    g.columns = [f"rfm_hist_{c[0]}_{c[1]}" for c in g.columns]

    # Historical patterns (safe to use)
    g["rfm_hist_monthly_orders"] = g.rfm_hist_order_purchase_timestamp_count / window_months
    g["rfm_hist_shipping_ratio"] = (g.rfm_hist_freight_value_sum / 
                                   g.rfm_hist_price_sum.replace({0: np.nan})).fillna(0)
    
    # Customer segments based on historical behavior
    if is_train:
        value_thresh = g.rfm_hist_payment_value_sum.quantile(0.75)
        freq_thresh = g.rfm_hist_monthly_orders.quantile(0.75)
        thresholds = {"hist_value": value_thresh, "hist_freq": freq_thresh}
    else:
        if thresholds is None:
            thresholds = {"hist_value": 50, "hist_freq": 1}
        value_thresh = thresholds["hist_value"]
        freq_thresh = thresholds["hist_freq"]
    
    g["rfm_hist_was_valuable"] = (g.rfm_hist_payment_value_sum > value_thresh).astype(int)
    g["rfm_hist_was_frequent"] = (g.rfm_hist_monthly_orders > freq_thresh).astype(int)
    
    return g[["rfm_hist_monthly_orders", "rfm_hist_payment_value_mean", 
             "rfm_hist_shipping_ratio", "rfm_hist_was_valuable", 
             "rfm_hist_was_frequent"]], thresholds

    
def payment_features(payments: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    if payments.empty:
        return pd.DataFrame()
    
    # Order-level payment aggregation
    pc = payments.groupby("order_id").agg({
        "payment_installments": "max",
        "payment_type": "nunique",
        "payment_value": "sum",
    })
    
    # Customer-level features
    cpf = (
        orders[["order_id", "customer_id"]]
        .merge(pc.reset_index(), on="order_id", how="left")
        .groupby("customer_id")
        .agg({
            "payment_installments": "mean",
            "payment_type": "mean",
            "payment_value": "mean",
        })
        .round(2)
    )
    
    # Simple binary flags
    cpf["payment_high_installments"] = (cpf.payment_installments > 6).astype(int)
    cpf["payment_multiple_types"] = (cpf.payment_type > 1).astype(int)
    
    return cpf


def cancellation_features(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame()
    
    cf = orders.groupby("customer_id").agg(
        total_orders=("order_id","count"),
        canceled_orders=("order_status", lambda x: (x=="canceled").sum())
    )
    cf["cancellation_ratio"] = cf.canceled_orders / cf.total_orders
    cf["has_cancellations"] = (cf.canceled_orders > 0).astype(int)
    
    return cf[["cancellation_ratio", "has_cancellations"]]


def build_for(cust_ids, orders, reviews, items, payments,
              is_train=False, thresholds=None):
    if len(cust_ids) == 0:
        return pd.DataFrame(), thresholds
    
    o = orders[orders.customer_id.isin(cust_ids)]
    r = reviews[reviews.order_id.isin(o.order_id)]
    i = items[items.order_id.isin(o.order_id)]
    p = payments[payments.order_id.isin(o.order_id)]

    D = delivery_features(o)
    V = review_features(r, o)
    R, thresholds = rfm_features_historical_window(o, i, p, is_train, thresholds)
    P = payment_features(p, o)
    C = cancellation_features(o)

    df = pd.DataFrame({"customer_id": cust_ids}).set_index("customer_id")
    for X in (D, V, R, P, C):
        if not X.empty:
            df = df.join(X, how="left")
    return df.fillna(0).reset_index(), thresholds

# ─────────────────────────────────────────────────────────────────────────────
# TASKS
# ─────────────────────────────────────────────────────────────────────────────
@task(name="load_data", retries=2, retry_delay_seconds=5)
def load_data():
    """Load processed Olist tables + basic datetime coercion."""
    paths = {
        "orders": "olist_orders_dataset.parquet",
        "reviews": "olist_order_reviews_dataset.parquet",
        "items": "olist_order_items_dataset.parquet",
        "payments": "olist_order_payments_dataset.parquet",
        "customers": "olist_customers_dataset.parquet",
    }
    dfs = {k: pd.read_parquet(PROCESSED_DATA_DIR / v) for k, v in paths.items()}

    # minimal datetime parsing (orders has the most)
    dt_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in dt_cols:
        if col in dfs["orders"].columns:
            dfs["orders"][col] = pd.to_datetime(dfs["orders"][col], errors="coerce")

    if "review_creation_date" in dfs["reviews"].columns:
        dfs["reviews"]["review_creation_date"] = pd.to_datetime(
            dfs["reviews"]["review_creation_date"], errors="coerce")

    if "shipping_limit_date" in dfs["items"].columns:
        dfs["items"]["shipping_limit_date"] = pd.to_datetime(
            dfs["items"]["shipping_limit_date"], errors="coerce")

    return dfs


@task(name="define_labels")
def define_labels(orders: pd.DataFrame) -> pd.DataFrame:
    latest = orders["order_purchase_timestamp"].max()
    cutoff = latest - DateOffset(months=CHURN_MONTHS)

    last = orders.groupby("customer_id").order_purchase_timestamp.max().reset_index()
    last.rename(columns={"order_purchase_timestamp": "last_purchase"}, inplace=True)
    last["is_churned"] = (last.last_purchase < cutoff).astype(int)
    last["days_since_last_order"] = (latest - last.last_purchase).dt.days
    return last


@task(name="split_ids")
def split_ids(labels: pd.DataFrame):
    train_val, test = train_test_split(
        labels, test_size=TEST_SIZE, stratify=labels.is_churned, random_state=SEED)
    rel_val = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val.is_churned, random_state=SEED)
    return train, val, test


@task(name="build_and_save", retries=1, retry_delay_seconds=5)
def build_and_save(train_ids, val_ids, test_ids, dfs: dict[str, pd.DataFrame]):
    logger = get_run_logger()

    train_df, thr = build_for(train_ids.customer_id, dfs["orders"], dfs["reviews"],
                              dfs["items"], dfs["payments"], is_train=True)
    val_df, _   = build_for(val_ids.customer_id, dfs["orders"], dfs["reviews"],
                            dfs["items"], dfs["payments"], is_train=False, thresholds=thr)
    test_df, _  = build_for(test_ids.customer_id, dfs["orders"], dfs["reviews"],
                            dfs["items"], dfs["payments"], is_train=False, thresholds=thr)

    # attach labels
    train_df = train_df.merge(train_ids[["customer_id", "is_churned", "days_since_last_order"]], 
                             on="customer_id", how="left")
    val_df = val_df.merge(val_ids[["customer_id", "is_churned", "days_since_last_order"]], 
                         on="customer_id", how="left")
    test_df = test_df.merge(test_ids[["customer_id", "is_churned", "days_since_last_order"]], 
                           on="customer_id", how="left")

    feat_cols = [c for c in train_df.columns if c not in ("customer_id", "is_churned", "days_since_last_order")]
    logger.info("Built %s features", len(feat_cols))

    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        file = PROCESSED_DATA_DIR / f"churn_{name}_seed{SEED}.parquet"
        df.to_parquet(file, index=False)
        logger.info("💾 saved %s (rows=%s)", file.name, len(df))


# ─────────────────────────────────────────────────────────────────────────────
# FLOW
# ─────────────────────────────────────────────────────────────────────────────
@flow(name="build_customer_churn_master")
def build_customer_churn_master():
    log = get_run_logger()
    log.info("🚀 Starting churn‑dataset builder flow")

    dfs = load_data()
    labels = define_labels(dfs["orders"])
    train_ids, val_ids, test_ids = split_ids(labels)
    build_and_save(train_ids, val_ids, test_ids, dfs)

    log.info("✅ churn dataset (train/val/test) ready!")


if __name__ == "__main__":
    build_customer_churn_master()