"""
Prefect flow that loads cleaned Parquet tables (data/processed/) into the
Postgres service defined in infra/docker-compose.yml.

Run once:
    python -m flows.load_cleaned_to_db
Or:
    make db-load
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final, List

import pandas as pd
from sqlalchemy import create_engine
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta

# ───────────────────────────────────────────────
# Environment & paths
# ───────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = (
    Path(os.getenv("PROCESSED_DATA_DIR"))
    if os.getenv("PROCESSED_DATA_DIR")
    else PROJECT_ROOT / "data" / "processed"
)

# ───────────────────────────────────────────────
# Connection
# ───────────────────────────────────────────────
PG_USER = os.getenv("PG_USER", "olist")
PG_PW = os.getenv("PG_PASSWORD", "olist")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB", "olist")

POSTGRES_URL: Final = f"postgresql://{PG_USER}:{PG_PW}@{PG_HOST}:{PG_PORT}/{PG_DB}"
engine = create_engine(POSTGRES_URL, pool_pre_ping=True)

# ───────────────────────────────────────────────
# Table map
# ───────────────────────────────────────────────
TABLE_MAP: Final[dict[str, str]] = {
    "orders_clean.parquet": "orders_clean",
    "customers_clean.parquet": "customers_clean",
    "order_items_clean.parquet": "order_items_clean",
    "order_payments_clean.parquet": "order_payments_clean",
    "order_reviews_clean.parquet": "order_reviews_clean",
    "sellers_clean.parquet": "sellers_clean",
    "products_clean.parquet": "products_clean",
    "geolocation_clean.parquet": "geolocation_clean",
    "category_translation_clean.parquet": "category_translation_clean",
}

# ───────────────────────────────────────────────
# Tasks
# ───────────────────────────────────────────────
@task(
    name="load_table",
    retries=2,
    retry_delay_seconds=5,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
)
def load_table(parquet_name: str, table_name: str) -> None:
    log = get_run_logger()
    path = PROCESSED_DATA_DIR / parquet_name
    df = pd.read_parquet(path)

    # warn if generic object cols remain
    orphan_objects = [
        col
        for col, dtype in df.dtypes.items()
        if dtype == "object" and not col.endswith("_id")
    ]
    if orphan_objects:
        log.debug("Generic 'object' columns in %s: %s", table_name, orphan_objects)

    df.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False,
        chunksize=10_000,
        method="multi",
    )
    log.info("Loaded %d rows → %s", len(df), table_name)


@task(name="schema_check")
def schema_check(table_name: str, expected_cols: List[str]) -> None:
    log = get_run_logger()
    cols = pd.read_sql(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %(tbl)s
        ORDER BY ordinal_position;
        """,
        engine,
        params=dict(tbl=table_name),
    )["column_name"].tolist()

    missing = [c for c in expected_cols if c not in cols]
    if missing:
        raise ValueError(f"{table_name}: missing columns {missing}")
    log.info("Schema OK → %s", table_name)


# ───────────────────────────────────────────────
# Flow
# ───────────────────────────────────────────────
@flow(name="load_cleaned_to_postgres")
def load_flow() -> None:
    logger = get_run_logger()

    futures = {}
    # kick off all load tasks concurrently
    for parquet_name, table in TABLE_MAP.items():
        futures[table] = load_table.submit(parquet_name, table)

    # example schema guard – wait for orders load to finish first
    orders_future = futures["orders_clean"]
    schema_check.submit(
        "orders_clean",
        ["order_id", "customer_id", "order_purchase_timestamp"],
        wait_for=[orders_future],
    )

    # wait for every task (loads + schema check) to finish
    for fut in futures.values():
        fut.result()

    logger.info("✅  All cleaned tables loaded into Postgres")


if __name__ == "__main__":
    load_flow()