"""Prefect 2.x flow that cleans every Olist raw CSV and writes typed Parquet
files to ``data/processed``. The logic here mirrors the thorough steps in
notebooks/0.0-initial-data-cleaning.ipynb

Run locally:
    python -m flows.clean_olist

Or via Makefile target:
    make clean-data
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
from prefect import flow, get_run_logger, task
import unidecode

# ---------------------------------------------------------------------------
# Paths (taken from project‑level config; fallback to relative paths)
# ---------------------------------------------------------------------------
try:
    from customer_ai.config import PROCESSED_DATA_DIR, RAW_DATA_DIR  # type: ignore
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save_clean(df: pd.DataFrame, name: str) -> Path:
    """Write dataframe to Parquet in processed dir and return path."""
    out_path = PROCESSED_DATA_DIR / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Task: Orders
# ---------------------------------------------------------------------------


@task(retries=1, retry_delay_seconds=5, name="clean_orders")
def clean_orders() -> Path:
    logger = get_run_logger()
    df = pd.read_csv(RAW_DATA_DIR / "olist_orders_dataset.csv")
    df = df.drop_duplicates(subset="order_id")

    ts_cols: Final[list[str]] = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    df[ts_cols] = df[ts_cols].apply(pd.to_datetime, errors="coerce")

    df["delivery_time_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days
    # replace negatives with NaN so they won’t trigger modelling errors later
    neg_mask = df["delivery_time_days"] < 0
    if neg_mask.any():
        logger.warning(
            "Found %d negative delivery_time_days → set to NaN", neg_mask.sum()
        )
        df.loc[neg_mask, "delivery_time_days"] = pd.NA

    # sanity assert AFTER patch
    assert (df["delivery_time_days"] >= 0).all() or df[
        "delivery_time_days"
    ].isna().any(), "Still negative delivery days after patch!"

    path = save_clean(df, "orders_clean")
    logger.info("Saved %d cleaned orders → %s", len(df), path)
    return path


# ---------------------------------------------------------------------------
# Task: Customers
# ---------------------------------------------------------------------------


@task(retries=1, retry_delay_seconds=5, name="clean_customers")
def clean_customers() -> Path:
    logger = get_run_logger()
    df = pd.read_csv(RAW_DATA_DIR / "olist_customers_dataset.csv")

    df = df.drop_duplicates(subset="customer_id")
    df["customer_zip_code_prefix"] = (
        df["customer_zip_code_prefix"].astype(str).str.zfill(5)
    )
    df["customer_city"] = (
        df["customer_city"].str.strip().str.lower().map(unidecode.unidecode)
    )
    df["customer_state"] = df["customer_state"].str.upper().astype("category")

    # merge lat/lon from geolocation table (pre‑merge, so raw geo file used)
    geo = pd.read_csv(RAW_DATA_DIR / "olist_geolocation_dataset.csv")
    geo = geo.drop_duplicates(subset="geolocation_zip_code_prefix").rename(
        columns={
            "geolocation_zip_code_prefix": "customer_zip_code_prefix",
            "geolocation_lat": "latitude",
            "geolocation_lng": "longitude",
        }
    )
    geo["customer_zip_code_prefix"] = (
        geo["customer_zip_code_prefix"].astype(str).str.zfill(5)
    )
    df = df.merge(
        geo[["customer_zip_code_prefix", "latitude", "longitude"]],
        on="customer_zip_code_prefix",
        how="left",
    )

    path = save_clean(df, "customers_clean")
    logger.info("Saved %d cleaned customers → %s", len(df), path)
    return path


# ---------------------------------------------------------------------------
# Task: Order Items
# ---------------------------------------------------------------------------


@task(retries=1, retry_delay_seconds=5, name="clean_order_items")
def clean_order_items() -> Path:
    logger = get_run_logger()
    df = pd.read_csv(RAW_DATA_DIR / "olist_order_items_dataset.csv")

    df = df.drop_duplicates(subset=["order_id", "order_item_id"])
    df["shipping_limit_date"] = pd.to_datetime(
        df["shipping_limit_date"], errors="coerce"
    )
    df["price"] = df["price"].astype(float)
    df["freight_value"] = df["freight_value"].astype(float)
    df["total_cost"] = df["price"] + df["freight_value"]

    path = save_clean(df, "order_items_clean")
    logger.info("Saved %d cleaned order items → %s", len(df), path)
    return path


# ---------------------------------------------------------------------------
# Task: Payments
# ---------------------------------------------------------------------------


@task(retries=1, retry_delay_seconds=5, name="clean_payments")
def clean_payments() -> Path:
    logger = get_run_logger()
    df = pd.read_csv(RAW_DATA_DIR / "olist_order_payments_dataset.csv")

    df = df.drop_duplicates(subset=["order_id", "payment_sequential"])
    df["payment_installments"] = df["payment_installments"].astype(int)
    df["payment_value"] = df["payment_value"].astype(float)

    path = save_clean(df, "order_payments_clean")
    logger.info("Saved %d cleaned payments → %s", len(df), path)
    return path


# ---------------------------------------------------------------------------
# Task: Reviews
# ---------------------------------------------------------------------------


@task(retries=1, retry_delay_seconds=5, name="clean_reviews")
def clean_reviews() -> Path:
    logger = get_run_logger()
    df = pd.read_csv(RAW_DATA_DIR / "olist_order_reviews_dataset.csv")

    df = df.drop_duplicates(subset="review_id")
    df["review_creation_date"] = pd.to_datetime(
        df["review_creation_date"], errors="coerce"
    )
    df["review_answer_timestamp"] = pd.to_datetime(
        df["review_answer_timestamp"], errors="coerce"
    )
    df["response_time_days"] = (
        df["review_answer_timestamp"] - df["review_creation_date"]
    ).dt.days

    df["review_comment_title"] = df["review_comment_title"].fillna("")
    df["review_comment_message"] = df["review_comment_message"].fillna("")

    path = save_clean(df, "order_reviews_clean")
    logger.info("Saved %d cleaned reviews → %s", len(df), path)
    return path


# ---------------------------------------------------------------------------
# Task: Sellers
# ---------------------------------------------------------------------------


@task(retries=1, retry_delay_seconds=5, name="clean_sellers")
def clean_sellers() -> Path:
    logger = get_run_logger()
    df = pd.read_csv(RAW_DATA_DIR / "olist_sellers_dataset.csv")

    df = df.drop_duplicates(subset="seller_id")
    df["seller_zip_code_prefix"] = df["seller_zip_code_prefix"].astype(str).str.zfill(8)
    df["seller_city"] = (
        df["seller_city"].str.strip().str.lower().map(unidecode.unidecode)
    )
    df["seller_state"] = df["seller_state"].str.upper().astype("category")

    path = save_clean(df, "sellers_clean")
    logger.info("Saved %d cleaned sellers → %s", len(df), path)
    return path


# ---------------------------------------------------------------------------
# Task: Products
# ---------------------------------------------------------------------------


@task(retries=1, retry_delay_seconds=5, name="clean_products")
def clean_products() -> Path:
    logger = get_run_logger()
    df = pd.read_csv(RAW_DATA_DIR / "olist_products_dataset.csv")

    df = df.drop_duplicates(subset="product_id")
    numeric_cols = [
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df["product_volume_cm3"] = (
        df["product_length_cm"] * df["product_width_cm"] * df["product_height_cm"]
    )
    df["product_density_g_cm3"] = df["product_weight_g"] / df["product_volume_cm3"]

    # merge category translations
    cat = pd.read_csv(RAW_DATA_DIR / "product_category_name_translation.csv")
    cat = cat.rename(
        columns={
            "product_category_name": "category_br",
            "product_category_name_english": "category_en",
        }
    )[["category_br", "category_en"]]
    df = df.merge(
        cat, left_on="product_category_name", right_on="category_br", how="left"
    )
    df = df.drop(columns=["product_category_name", "category_br"])

    path = save_clean(df, "products_clean")
    logger.info("Saved %d cleaned products → %s", len(df), path)
    return path


# ---------------------------------------------------------------------------
# Task: Geolocation
# ---------------------------------------------------------------------------


@task(retries=1, retry_delay_seconds=5, name="clean_geolocation")
def clean_geolocation() -> Path:
    logger = get_run_logger()
    df = pd.read_csv(RAW_DATA_DIR / "olist_geolocation_dataset.csv")

    df = df.drop_duplicates(subset="geolocation_zip_code_prefix")
    df["geolocation_zip_code_prefix"] = (
        df["geolocation_zip_code_prefix"].astype(str).str.zfill(8)
    )
    df = df.rename(
        columns={
            "geolocation_zip_code_prefix": "zip_code_prefix",
            "geolocation_lat": "latitude",
            "geolocation_lng": "longitude",
            "geolocation_city": "city",
            "geolocation_state": "state",
        }
    )

    path = save_clean(df, "geolocation_clean")
    logger.info("Saved %d cleaned geolocation rows → %s", len(df), path)
    return path


# ---------------------------------------------------------------------------
# Task: Category translation
# ---------------------------------------------------------------------------


@task(retries=1, retry_delay_seconds=5, name="clean_category_translation")
def clean_category_translation() -> Path:
    logger = get_run_logger()
    df = pd.read_csv(RAW_DATA_DIR / "product_category_name_translation.csv", dtype=str)

    df = df.drop_duplicates(subset="product_category_name")
    df = df.rename(
        columns={
            "product_category_name": "category_br",
            "product_category_name_english": "category_en",
        }
    )

    path = save_clean(df, "category_translation_clean")
    logger.info("Saved %d category translations → %s", len(df), path)
    return path


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


@flow(name="clean_olist", retries=0)
def clean_olist_flow():
    """Run all cleaning tasks sequentially so dependencies are respected."""

    _ = clean_orders()
    _ = clean_customers()
    _ = clean_order_items()
    _ = clean_payments()
    _ = clean_reviews()
    _ = clean_sellers()
    _ = clean_products()
    _ = clean_geolocation()
    _ = clean_category_translation()

    logger = get_run_logger()
    logger.info("All cleaned tables saved to %s", PROCESSED_DATA_DIR)


if __name__ == "__main__":
    clean_olist_flow()
