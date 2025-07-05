#!/usr/bin/env python
"""
Prefect wrapper: orchestrates the heavy script so training is tracked
in Prefect UI / can be scheduled.

Run:
    prefect server start     # (or use Prefect Cloud)
    python flows/build_customer_churn.py
"""
from __future__ import annotations
import sys
from pathlib import Path

from prefect import flow, task, get_run_logger

# add repo-root so we can import scripts.build_customer_churn ------------------
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJ_ROOT))

# ───────────────────────── Prefect tasks ────────────────────────────────
@task(retries=1, retry_delay_seconds=10, name="run_churn_pipeline")
def run_pipeline():
    log = get_run_logger()
    log.info("📦 importing training script …")
    from scripts.build_customer_churn import main   # noqa:  WPS433 (runtime import)
    res = main()
    log.info("✅ finished – test AUC %.3f", res["test_auc"])
    return res


@task(name="post_cleanup")
def cleanup(res: dict):
    log = get_run_logger()
    log.info("Artifacts for %s ready (F1 %.3f)", res["model_name"], res["test_f1"])


@flow(name="build_customer_churn_models")
def build_customer_churn_models():
    res = run_pipeline()
    cleanup(res)

# ───────────────────────── CLI ──────────────────────────────────────────
if __name__ == "__main__":
    build_customer_churn_models()