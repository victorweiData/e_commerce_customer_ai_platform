"""
Ingest the public **Olist Brazilian E-commerce** dataset:

1. Download the Kaggle ZIP into data/raw/
2. Unzip all nine CSV files               → data/raw/
3. Convert every CSV to Parquet           → data/processed/

Run once:
    python -m flows.ingest_olist

Or via Makefile:
    make ingest-olist
"""
from __future__ import annotations

from pathlib import Path
from typing import Final, List
import zipfile

import pandas as pd
from prefect import flow, get_run_logger, task

from customer_ai.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# ───────────────────────────────────────────────
DATASET: Final[str] = "olistbr/brazilian-ecommerce"
ZIP_NAME: Final[str] = "brazilian-ecommerce.zip"
CSV_FILES: Final[List[str]] = [
    "olist_orders_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
]
# ───────────────────────────────────────────────


@task(retries=2, retry_delay_seconds=10, name="download_zip")
def download_zip() -> Path:
    """
    Download the Kaggle archive → data/raw/.
    Skips if the ZIP already exists locally.
    """
    log = get_run_logger()
    zip_path = RAW_DATA_DIR / ZIP_NAME
    if zip_path.exists():
        log.info("📦 ZIP already present – skip download.")
        return zip_path

    try:
        from kaggle import api
    except ImportError as e:
        raise ImportError("Install the Kaggle CLI:  pip install kaggle") from e

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    api.authenticate()

    log.info(f"⬇️  Downloading {DATASET} …")
    api.dataset_download_files(
        DATASET,
        path=str(RAW_DATA_DIR),
        quiet=False,
        force=False,
    )

    # Kaggle saves as <slug>.zip – rename for consistency
    downloaded = next(RAW_DATA_DIR.glob("*.zip"))
    downloaded.rename(zip_path)
    log.info(f"ZIP saved → {zip_path}")
    return zip_path


@task(name="unzip_csvs")
def unzip_csvs(zip_path: Path) -> List[Path]:
    """
    Extract the nine Olist CSVs into data/raw/.
    Returns their paths.
    """
    log = get_run_logger()
    csv_paths: List[Path] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for csv in CSV_FILES:
            out = RAW_DATA_DIR / csv
            if out.exists():
                log.info(f"🗃️  {csv} already extracted – skip.")
            else:
                log.info(f"🗃️  Extracting {csv}")
                zf.extract(csv, path=RAW_DATA_DIR)
            csv_paths.append(out)

    log.info("All CSVs extracted.")
    return csv_paths


@task(name="csv_to_parquet", retries=1, retry_delay_seconds=5)
def csv_to_parquet(csv_paths: List[Path]) -> List[Path]:
    """
    Convert each CSV → Parquet into data/processed/.
    """
    log = get_run_logger()
    pq_paths: List[Path] = []
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for csv in csv_paths:
        pq = PROCESSED_DATA_DIR / (csv.stem + ".parquet")
        if pq.exists():
            log.info(f"🟢 {pq.name} already exists – skip convert.")
        else:
            log.info(f"🔄 {csv.name} → Parquet")
            df = pd.read_csv(csv)
            df.to_parquet(pq, index=False)
        pq_paths.append(pq)

    log.info("CSV → Parquet conversion finished.")
    return pq_paths


# ───────────────────────────────────────────────
@flow(name="ingest_olist")
def ingest_flow() -> List[Path]:
    """
    Orchestration: download → unzip → convert.
    Returns list of Parquet paths.
    """
    zip_path = download_zip()
    csv_paths = unzip_csvs(zip_path)
    parquet_paths = csv_to_parquet(csv_paths)
    return parquet_paths


if __name__ == "__main__":
    ingest_flow()
