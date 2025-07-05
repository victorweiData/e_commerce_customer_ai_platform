"""
Prefect flow to load original Olist Parquet files from `data/processed/`
into the Postgres DB defined in `infra/docker-compose.yml`.

Run once:
    python -m flows.load_raw_to_db
Or:
    make db-load-raw
"""
from __future__ import annotations

from datetime import timedelta
import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.tasks import task_input_hash
from sqlalchemy import create_engine

# ────────────── Environment & Paths ──────────────
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ────────────── DB Connection ──────────────
PG_USER = os.getenv("PG_USER", "olist")
PG_PW = os.getenv("PG_PASSWORD", "olist")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB", "olist")

POSTGRES_URL: Final = f"postgresql://{PG_USER}:{PG_PW}@{PG_HOST}:{PG_PORT}/{PG_DB}"
engine = create_engine(POSTGRES_URL, pool_pre_ping=True)

# ────────────── File-to-Table Map ──────────────
TABLE_MAP: Final[dict[str, str]] = {
    "olist_orders_dataset.parquet": "orders_raw",
    "olist_customers_dataset.parquet": "customers_raw",
    "olist_order_items_dataset.parquet": "order_items_raw",
    "olist_order_payments_dataset.parquet": "order_payments_raw",
    "olist_order_reviews_dataset.parquet": "order_reviews_raw",
    "olist_sellers_dataset.parquet": "sellers_raw",
    "olist_products_dataset.parquet": "products_raw",
    "olist_geolocation_dataset.parquet": "geolocation_raw",
    "product_category_name_translation.parquet": "category_translation_raw",
}

# ────────────── Tasks ──────────────
@task(
    name="load_parquet_to_postgres",
    retries=2,
    retry_delay_seconds=5,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
)
def load_table(parquet_file: str, table_name: str) -> None:
    log = get_run_logger()
    df = pd.read_parquet(DATA_DIR / parquet_file)

    df.to_sql(
        table_name,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=10_000,
        method="multi"
    )

    log.info("✅ Loaded %d rows into %s", len(df), table_name)

# ────────────── Flow ──────────────
@flow(name="load_processed_parquet_to_postgres")
def load_processed_flow() -> None:
    logger = get_run_logger()

    futures = {
        name: load_table.submit(parquet, name)
        for parquet, name in TABLE_MAP.items()
    }

    for fut in futures.values():
        fut.result()

    logger.info("✅ All processed Parquet tables loaded into Postgres")

if __name__ == "__main__":
    load_processed_flow()