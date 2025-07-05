"""
Download all processed Olist tables, build the customer_master table
(with RFM, monetary, payment, review, category & geo features) and
write it to data/processed/customer_master.parquet.

Same Procedure as notebooks/1.0-victorwei-modeling-clustering-customer-segmentaion.ipynb

Run:
    python -m flows.build_customer_segments_master       # or: make ingest-master
"""
# build_customer_segments_master.py

from __future__ import annotations

from pathlib import Path
import pandas as pd
from functools import reduce

from prefect import flow, task, get_run_logger

from customer_ai.config import PROCESSED_DATA_DIR

# list of processed files to load
TABLE_FILES = {
    "orders"    : "olist_orders_dataset.parquet",
    "items"     : "olist_order_items_dataset.parquet",
    "payments"  : "olist_order_payments_dataset.parquet",
    "reviews"   : "olist_order_reviews_dataset.parquet",
    "customers" : "olist_customers_dataset.parquet",
    "products"  : "olist_products_dataset.parquet",
    "geoloc"    : "olist_geolocation_dataset.parquet",
    "cats"      : "product_category_name_translation.parquet",
}

@task
def load_tables() -> dict[str, pd.DataFrame]:
    log = get_run_logger()
    dfs = {}
    for name, fname in TABLE_FILES.items():
        path = PROCESSED_DATA_DIR / fname
        log.info(f"Loading {name} ← {path}")
        dfs[name] = pd.read_parquet(path)
    return dfs

@task
def build_master(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    log = get_run_logger()
    orders    = dfs["orders"].copy()
    items     = dfs["items"]
    payments  = dfs["payments"]
    reviews   = dfs["reviews"]
    customers = dfs["customers"]
    products  = dfs["products"]
    cats      = dfs["cats"]

    # ── 1) RFM metrics ──────────────────────────────────────────────────────
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    analysis_date = orders["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = (
        orders
          .groupby("customer_id")["order_purchase_timestamp"]
          .agg(first_order="min", last_order="max", frequency="count")
    )
    rfm["recency_days"] = (analysis_date - rfm["last_order"]).dt.days
    rfm["tenure_days"]  = (rfm["last_order"] - rfm["first_order"]).dt.days
    rfm = rfm.reset_index()

    # ── 2) Monetary ─────────────────────────────────────────────────────────
    order_vals = (
        items.groupby("order_id")
             .agg(total_price    = ("price",         "sum"),
                  total_freight  = ("freight_value", "sum"))
             .assign(total_value=lambda d: d["total_price"] + d["total_freight"])
             .reset_index()
    )
    ord_cust = orders[["order_id","customer_id"]].merge(order_vals, on="order_id")
    monetary = (
        ord_cust.groupby("customer_id")
                .agg(total_spent     = ("total_value", "sum"),
                     avg_order_value = ("total_value", "mean"),
                     total_freight   = ("total_freight","sum"),
                     avg_freight     = ("total_freight","mean"))
                .reset_index()
    )

    # ── 3) Payment behavior ──────────────────────────────────────────────────
    pay_cust = (
        payments[["order_id","payment_type","payment_installments","payment_value"]]
          .merge(orders[["order_id","customer_id"]], on="order_id", how="left")
    )
    payment_behavior = (
        pay_cust.groupby("customer_id")
                .agg(
                    preferred_payment_type = ("payment_type",       lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"),
                    avg_installments       = ("payment_installments","mean"),
                    total_payment_value    = ("payment_value",      "sum"),
                    payment_diversity      = ("payment_type",       "nunique")
                )
                .round(2)
                .reset_index()
    )

    # ── 4) Review behavior ───────────────────────────────────────────────────
    rev_cust = orders[["order_id","customer_id"]].merge(reviews, on="order_id", how="left")
    review_behavior = (
        rev_cust.groupby("customer_id")
                .agg(
                    avg_review_score  = ("review_score",         "mean"),
                    review_count      = ("review_score",         "count"),
                    reviews_with_text = ("review_comment_message", lambda x: x.notna().sum())
                )
                .reset_index()
    )
    review_behavior = (
        review_behavior
          .merge(rfm[["customer_id","frequency"]], on="customer_id", how="left")
          .assign(review_rate=lambda df: (df["review_count"]/df["frequency"]).fillna(0).round(2))
    )

    # ── 5) Category behavior ────────────────────────────────────────────────
    items_with_cats = (
        items.merge(products[["product_id","product_category_name"]], on="product_id", how="left")
             .merge(cats, on="product_category_name", how="left")
             .merge(orders[["order_id","customer_id"]], on="order_id", how="left")
    )
    category_behavior = (
        items_with_cats.groupby("customer_id")
                        .agg(
                            category_diversity = ("product_category_name_english","nunique"),
                            preferred_category = ("product_category_name_english", lambda s: s.mode().iat[0] if not s.mode().empty else "other")
                        )
                        .reset_index()
    )

    # ── 6) Geographic info ─────────────────────────────────────────────────
    geo_info = customers[["customer_id","customer_city","customer_state"]]

    # ── 7) Merge everything ────────────────────────────────────────────────
    dfs_to_merge = [
        rfm, monetary, payment_behavior,
        review_behavior.drop(columns="frequency", errors="ignore"),
        category_behavior, geo_info
    ]
    master = reduce(lambda a,b: a.merge(b, on="customer_id", how="left"), dfs_to_merge)
    master = master.fillna(0)

    # ── 8) RFM quintile scores & value segments ─────────────────────────────
    # R score: smaller recency → higher bucket (5)
    master["R_score"] = pd.qcut(master["recency_days"], q=5, labels=[5,4,3,2,1]).astype(int)
    master["F_score"] = pd.qcut(master["frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5]).astype(int)
    master["M_score"] = pd.qcut(master["total_spent"].rank(method="first"), q=5, labels=[1,2,3,4,5]).astype(int)
    master["RFM_score"] = master["R_score"].astype(str) + master["F_score"].astype(str) + master["M_score"].astype(str)

    quantiles = master["total_spent"].quantile([.25,.5,.75,.9])
    def bucket(x):
        if x>=quantiles[.9] : return "VIP"
        if x>=quantiles[.75]: return "High Value"
        if x>=quantiles[.5] : return "Medium Value"
        if x>=quantiles[.25]: return "Low Value"
        return "Entry Level"
    master["value_segment"] = master["total_spent"].apply(bucket)

    log.info("Built customer_master with %d customers × %d features", *master.shape)
    
    # Ensure all object columns are strings for Parquet compatibility
    for col in master.select_dtypes(include=['object']).columns:
        master[col] = master[col].astype(str)
    
    return master

@task
def save_master(master: pd.DataFrame) -> Path:
    out = PROCESSED_DATA_DIR / "customer_master.parquet"
    master.to_parquet(out, index=False)
    get_run_logger().info("Wrote customer_master → %s", out)
    return out

@flow(name="build_customer_master")
def build_customer_master():
    dfs    = load_tables()
    master = build_master(dfs)
    save_master(master)

if __name__ == "__main__":
    build_customer_master()