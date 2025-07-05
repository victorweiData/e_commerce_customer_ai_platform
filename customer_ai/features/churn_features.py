"""customer_ai.features.churn_features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utility module that houses **all feature engineering logic** for the churn
pipeline.

Keeping this code separate lets flows (and notebooks) import it without
circular references.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

try:
    from textblob import TextBlob  # type: ignore
except ImportError:  # pragma: no cover
    TextBlob = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS / CONFIG
# ──────────────────────────────────────────────────────────────────────────────
HIST_WINDOW_MONTHS = 6  # window length **and** gap before churn horizon

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def delivery_features(orders: pd.DataFrame) -> pd.DataFrame:
    """On-time vs. slow delivery behaviour and derived flags."""
    df = (orders.query("order_status == 'delivered'")
                 .dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date"])
                 .copy())
    if df.empty:
        return pd.DataFrame()

    df["days_diff"]   = (df.order_delivered_customer_date - df.order_estimated_delivery_date).dt.days
    df["actual_days"] = (df.order_delivered_customer_date - df.order_purchase_timestamp).dt.days

    g = (df.groupby("customer_id")
            .agg(days_diff_mean=("days_diff", "mean"),
                 actual_days_mean=("actual_days", "mean"),
                 actual_days_std=("actual_days", "std"),
                 n_orders=("order_id", "count"))
            .round(2))

    # ratios and flags --------------------------------------------------------
    late = df[df.days_diff > 0].groupby("customer_id").size()
    slow = df[df.actual_days > 7].groupby("customer_id").size()
    vslow = df[df.actual_days > 14].groupby("customer_id").size()

    g["late_ratio"]       = (late / g.n_orders).fillna(0)
    g["slow_ratio"]       = (slow / g.n_orders).fillna(0)
    g["very_slow_ratio"]  = (vslow / g.n_orders).fillna(0)

    g["consistently_late"]    = (g.late_ratio > 0.5).astype(int)
    g["frequently_slow"]      = (g.slow_ratio > 0.3).astype(int)
    g["has_very_slow"]        = (g.very_slow_ratio > 0).astype(int)
    g["inconsistent_delivery"] = (g.actual_days_std > 5).fillna(0).astype(int)

    g["dissatisfaction"] = (0.3 * g.late_ratio + 0.4 * g.slow_ratio + 0.3 * g.very_slow_ratio)
    g["likely_dissatisfied"] = (g.dissatisfaction > 0.3).astype(int)

    # speed buckets
    g["speed_fast"]        = (g.actual_days_mean <= 3).astype(int)
    g["speed_normal"]      = ((g.actual_days_mean > 3) & (g.actual_days_mean <= 7)).astype(int)
    g["speed_slow"]        = ((g.actual_days_mean > 7) & (g.actual_days_mean <= 14)).astype(int)
    g["speed_very_slow"]   = (g.actual_days_mean > 14).astype(int)

    g.drop(columns=["n_orders"], inplace=True)
    g.columns = ["delivery_" + c for c in g.columns]
    return g


def review_features(reviews: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    if reviews.empty:
        return pd.DataFrame()

    rv = reviews.dropna(subset=["review_score"]).copy()
    if rv.empty:
        return pd.DataFrame()

    rv = rv.merge(orders[["order_id", "customer_id"]], on="order_id", how="left")

    g = (rv.groupby("customer_id")
            .agg(review_score_mean=("review_score", "mean"),
                 review_score_min=("review_score", "min"),
                 n_reviews=("order_id", "count"))
            .round(2))

    bad = rv[rv.review_score <= 2].groupby("customer_id").size()
    g["bad_ratio"]        = (bad / g.n_reviews).fillna(0)
    g["dissatisfied"]     = (g.review_score_mean < 3.5).astype(int)
    g.drop(columns=["n_reviews"], inplace=True)

    # optional sentiment -----------------------------------------------------
    if TextBlob is not None and "review_comment_message" in reviews.columns:
        txt = rv.dropna(subset=["review_comment_message"])[["customer_id", "review_comment_message"]]
        if not txt.empty:
            txt["sentiment"] = txt.review_comment_message.apply(lambda t: TextBlob(t).sentiment.polarity)
            sent = txt.groupby("customer_id").sentiment.mean()
            g["sentiment_mean"] = sent
    g.columns = ["review_" + c for c in g.columns]
    return g


def _historical_slice(orders: pd.DataFrame, months_back: int) -> pd.DataFrame:
    latest = orders.order_purchase_timestamp.max()
    end = latest - DateOffset(months=months_back)
    start = end - DateOffset(months=months_back)
    return orders[(orders.order_purchase_timestamp >= start) & (orders.order_purchase_timestamp <= end)].copy()


def rfm_historical(orders: pd.DataFrame, items: pd.DataFrame, payments: pd.DataFrame,
                   *, is_train: bool, thresholds: dict | None) -> tuple[pd.DataFrame, dict]:
    win_orders = _historical_slice(orders, HIST_WINDOW_MONTHS)
    if win_orders.empty:
        return pd.DataFrame(), thresholds or {}

    od = win_orders.merge(items, on="order_id", how="left")
    pay_sum = payments.groupby("order_id").payment_value.sum().reset_index()
    od = od.merge(pay_sum, on="order_id", how="left")

    g = (od.groupby("customer_id")
            .agg(monthly_orders=("order_id", lambda s: len(s) / HIST_WINDOW_MONTHS),
                 spend_sum=("payment_value", "sum"),
                 spend_mean=("payment_value", "mean"),
                 freight_sum=("freight_value", "sum"),
                 price_sum=("price", "sum"))
            .round(2))

    g["shipping_ratio"] = (g.freight_sum / g.price_sum.replace({0: np.nan})).fillna(0)

    if is_train:
        thresholds = {
            "val_thresh": g.spend_sum.quantile(0.75),
            "freq_thresh": g.monthly_orders.quantile(0.75),
        }

    g["was_valuable"] = (g.spend_sum > thresholds["val_thresh"]).astype(int)
    g["was_frequent"] = (g.monthly_orders > thresholds["freq_thresh"]).astype(int)

    g = g[["monthly_orders", "spend_mean", "shipping_ratio", "was_valuable", "was_frequent"]]
    g.columns = ["rfm_hist_" + c for c in g.columns]
    return g, thresholds


def payment_features(payments: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-order payment info to customer level without object-column errors."""
    if payments.empty:
        return pd.DataFrame()

    # ── 1. Per-order aggregation ──────────────────────────────────────────
    pc = payments.groupby("order_id").agg(
        install_max=("payment_installments", "max"),
        install_mean=("payment_installments", "mean"),
        type_nunique=("payment_type", "nunique"),
        payment_mean=("payment_value", "mean"),
    ).reset_index()  # keep order_id for merge

    # ── 2. Merge with customer_id and aggregate ───────────────────────────
    pay_cust = orders[["order_id", "customer_id"]].merge(pc, on="order_id", how="left")

    cpf = (pay_cust.groupby("customer_id")
            .agg(install_max_mean=("install_max", "mean"),
                 install_mean_mean=("install_mean", "mean"),
                 type_nunique_mean=("type_nunique", "mean"),
                 payment_mean_mean=("payment_mean", "mean"))
            .round(2))

    # ── 3. Flags ──────────────────────────────────────────────────────────
    cpf["high_installments"] = (cpf.install_max_mean > 6).astype(int)
    cpf["multiple_types"]   = (cpf.type_nunique_mean > 1).astype(int)

    cpf.columns = ["payment_" + c for c in cpf.columns]
    return cpf


def cancellation_features(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame()
    cf = orders.groupby("customer_id").agg(
        n_orders=("order_id", "count"),
        n_cancel=("order_status", lambda x: (x == "canceled").sum()),
    )
    cf["ratio"] = (cf.n_cancel / cf.n_orders).fillna(0)
    cf["has_cancel"] = (cf.n_cancel > 0).astype(int)
    cf = cf[["ratio", "has_cancel"]]
    cf.columns = ["cancel_" + c for c in cf.columns]
    return cf

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_for(
    customer_ids: pd.Series,
    orders: pd.DataFrame,
    reviews: pd.DataFrame,
    items: pd.DataFrame,
    payments: pd.DataFrame,
    *,
    is_train: bool = False,
    thresholds: Dict | None = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Return a DataFrame of engineered features for the given customers.

    Parameters
    ----------
    customer_ids : Series of unique customer_id values
    orders, reviews, items, payments : the raw tables (already filtered by time)
    is_train     : whether we're on the train split (to learn quantile thresholds)
    thresholds   : pass previously learned thresholds when `is_train=False`
    """
    # sub‑select order universe for these customers ---------------------------
    o = orders[orders.customer_id.isin(customer_ids)]
    r = reviews[reviews.order_id.isin(o.order_id)]
    i = items[items.order_id.isin(o.order_id)]
    p = payments[payments.order_id.isin(o.order_id)]

    # feature blocks ----------------------------------------------------------
    D = delivery_features(o)
    V = review_features(r, o)
    R, thresholds = rfm_historical(o, i, p, is_train=is_train, thresholds=thresholds)
    P = payment_features(p, o)
    C = cancellation_features(o)

    # assemble ---------------------------------------------------------------
    df = pd.DataFrame({"customer_id": customer_ids}).set_index("customer_id")
    for blk in (D, V, R, P, C):
        if not blk.empty:
            df = df.join(blk, how="left")
    df.fillna(0, inplace=True)
    df.reset_index(inplace=True)
    return df, thresholds
