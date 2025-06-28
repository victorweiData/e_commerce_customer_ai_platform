# flows/ingest_olist.py
from pathlib import Path
import subprocess, zipfile
from prefect import flow, task

RAW_DIR = Path("data/raw")

@task(retries=2, retry_delay_seconds=10)
def download_olist() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download",
         "-d", "olistbr/brazilian-ecommerce",
         "-p", str(RAW_DIR)],
        check=True,
    )
    return next(RAW_DIR.glob("brazilian-ecommerce*.zip"))

@task
def extract(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()

@flow(name="ingest_olist")
def ingest_flow():
    zip_path = download_olist()
    extract(zip_path)
    print("DONE: Olist raw files ready in data/raw")

if __name__ == "__main__":
    ingest_flow()