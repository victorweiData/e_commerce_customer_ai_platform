# customer_ai/features.py

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROC = PROJECT_ROOT / "data" / "processed"

def build_sales_fact_with_churn() -> Path:
    orders    = pd.read_parquet(PROC / "orders_clean.parquet")
    items     = pd.read_parquet(PROC / "order_items_clean.parquet")
    payments  = pd.read_parquet(PROC / "order_payments_clean.parquet")
    reviews   = pd.read_parquet(PROC / "order_reviews_clean.parquet")
    customers = pd.read_parquet(PROC / "customers_clean.parquet")[
        ["customer_id", "customer_unique_id"]
    ]

    pay_agg = (
        payments
        .groupby("order_id", as_index=False)
        .agg(total_paid=("payment_value", "sum"),
             n_payments=("payment_sequential", "count"))
    )
    rev_agg = (
        reviews
        .groupby("order_id", as_index=False)
        .agg(avg_score=("review_score", "mean"),
             max_response_time=("response_time_days", "max"))
    )

    fact = (
        orders
        .merge(items, on="order_id", how="left")
        .merge(pay_agg, on="order_id", how="left")
        .merge(rev_agg, on="order_id", how="left")
        .merge(customers, on="customer_id", how="left")
    )

    ref_date   = fact["order_purchase_timestamp"].max()
    cutoff     = ref_date - pd.Timedelta(days=90)
    last_purch = (
        fact
        .groupby("customer_unique_id", as_index=False)
        ["order_purchase_timestamp"].max()
        .rename(columns={"order_purchase_timestamp": "last_purchase_date"})
    )
    last_purch["churn"] = (last_purch["last_purchase_date"] < cutoff).astype(int)

    fact = fact.merge(
        last_purch[["customer_unique_id", "last_purchase_date", "churn"]],
        on="customer_unique_id", how="left"
    )

    out = PROC / "sales_fact_with_churn.parquet"
    fact.to_parquet(out, index=False)
    return out


def build_order_features() -> Path:
    orders   = pd.read_parquet(PROC / "orders_clean.parquet")
    items    = pd.read_parquet(PROC / "order_items_clean.parquet")
    products = pd.read_parquet(PROC / "products_clean.parquet")[
        ["product_id", "product_volume_cm3", "product_density_g_cm3"]
    ]

    df = (
        orders
        .merge(items, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
    )
    df["late_delivery"] = (df["days_diff_estimate"] < 0).astype(int)

    feats = (
        df
        .groupby("order_id", as_index=False)
        .agg(
            late_delivery=("late_delivery", "max"),
            approval_lag_hours=("approval_lag_hours", "first"),
            purchase_month=("order_purchase_timestamp", lambda s: s.iloc[0].month),
            purchase_dow=("order_purchase_timestamp", lambda s: s.iloc[0].dayofweek),
            n_items=("order_item_id", "nunique"),
            order_price=("price", "sum"),
            order_freight=("freight_value", "sum"),
            avg_item_volume=("product_volume_cm3", "mean"),
            avg_item_density=("product_density_g_cm3", "mean"),
            delivery_time_days=("delivery_time_days", "first"),
            est_delta_days=("days_diff_estimate", "first"),
        )
    )

    out = PROC / "order_features.parquet"
    feats.to_parquet(out, index=False)
    return out


def build_churn_features() -> Path:
    fact = pd.read_parquet(PROC / "sales_fact_with_churn.parquet")
    DATE_REF = fact["order_purchase_timestamp"].max()

    rfm = (
        fact
        .groupby("customer_unique_id", as_index=False)
        .agg(
            recency_days=("order_purchase_timestamp", lambda s: (DATE_REF - s.max()).days),
            frequency=("order_id", "nunique"),
            monetary_value=("total_paid", "sum"),
            first_purchase=("order_purchase_timestamp", "min"),
        )
    )
    rfm["cust_age_days"] = (DATE_REF - rfm["first_purchase"]).dt.days

    behav = (
        fact
        .groupby("customer_unique_id", as_index=False)
        .agg(
            avg_delivery_time_days=("delivery_time_days", "mean"),
            avg_approval_lag_hours=("approval_lag_hours", "mean"),
            avg_estimate_delta=("days_diff_estimate", "mean"),
            avg_rating=("avg_score", "mean"),
            avg_payment_count=("n_payments", "mean"),
            avg_items_per_order=("order_item_id", "nunique"),
        )
    )

    churn = fact[["customer_unique_id", "churn"]].drop_duplicates()
    cust_full = rfm.merge(behav, on="customer_unique_id").merge(churn, on="customer_unique_id")

    out = PROC / "churn_features.parquet"
    cust_full.to_parquet(out, index=False)
    return out


def build_segment_features() -> Path:
    cust = pd.read_parquet(PROC / "churn_features.parquet")
    cols = [c for c in cust.columns if c not in {"customer_unique_id", "churn", "first_purchase"}]
    seg  = cust[["customer_unique_id"] + cols]
    out  = PROC / "segment_features.parquet"
    seg.to_parquet(out, index=False)
    return out


def build_segment_features_extended() -> Path:
    """Build extended segmentation features:
       geography one-hots, promo share, and tenure buckets."""
    # load
    fact        = pd.read_parquet(PROC / "sales_fact_with_churn.parquet")
    customers   = pd.read_parquet(PROC / "customers_clean.parquet")[
                     ["customer_unique_id", "customer_state"]
                 ]
    rfm         = pd.read_parquet(PROC / "churn_features.parquet")[
                     ["customer_unique_id", "cust_age_days"]
                 ]
    cust_full   = pd.read_parquet(PROC / "churn_features.parquet")

    # 1) Geography top-10 state one-hots
    fact = fact.merge(customers, on="customer_unique_id", how="left")
    top_states = fact["customer_state"].value_counts().head(10).index
    fact["state_grouped"] = np.where(
        fact["customer_state"].isin(top_states),
        fact["customer_state"],
        "OTHER"
    )
    geo = (
        pd.get_dummies(
            fact[["customer_unique_id", "state_grouped"]],
            columns=["state_grouped"],
            dtype=int,
        )
        .groupby("customer_unique_id", as_index=False)
        .max()
    )

    # 2) Promo-month share
    PROMO_MONTHS = [6, 11, 12]
    fact["is_promo"] = fact["order_purchase_timestamp"]\
        .dt.month.isin(PROMO_MONTHS).astype(int)
    promo = (
        fact
        .groupby("customer_unique_id", as_index=False)
        .agg(promo_share=("is_promo", "mean"))
    )

    # 3) Tenure bucket one-hots
    TENURE_BINS   = [0, 180, 365, 730, np.inf]
    TENURE_LABELS = ["<6m", "6-12m", "1-2y", ">2y"]
    rfm["tenure_bucket"] = pd.cut(
        rfm["cust_age_days"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
    )
    tenure = pd.get_dummies(
        rfm[["customer_unique_id", "tenure_bucket"]],
        columns=["tenure_bucket"],
        dtype=int,
    )

    # assemble
    seg_ext = (
        cust_full
        .drop(columns=["first_purchase", "churn"])
        .merge(geo,    on="customer_unique_id", how="left")
        .merge(promo,  on="customer_unique_id", how="left")
        .merge(tenure, on="customer_unique_id", how="left")
    )

    out = PROC / "segment_features_extended.parquet"
    seg_ext.to_parquet(out, index=False)
    return out


def build_cltv_summary() -> Path:
    """Build and save BG/NBD + Gamma-Gamma summary (R, F, T, M)."""
    cust = pd.read_parquet(PROC / "churn_features.parquet")

    cltv_summary = (
        cust[[
            "customer_unique_id",
            "frequency",
            "recency_days",
            "cust_age_days",
            "monetary_value",
        ]]
        .rename(columns={
            "recency_days":  "recency_cal",  # lifetimes naming
            "cust_age_days": "T",
        })
    )

    out = PROC / "cltv_summary.parquet"
    cltv_summary.to_parquet(out, index=False)
    return out