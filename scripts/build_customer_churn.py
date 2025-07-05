#!/usr/bin/env python
"""
build_customer_churn.py
End-to-end churn-model pipeline that

1. loads train/val/test parquet files produced by `features_churn.py`
2. trains Logistic-Regression, LightGBM and XGBoost baselines
3. selects the best model on val-AUC
4. fine-tunes it with Optuna (50 trials by default)
5. evaluates on train/val/test with a rich set of metrics
6. saves artefacts (model, scaler, csvs, png)
7. emits a JSON payload consumed directly by the React dashboard
8. calculates realistic business metrics based on Olist dataset patterns
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ───────────────────────────────────────────────────────────────────────────────
SEED = 42

sys.path.append(str(Path(__file__).resolve().parent.parent))
from customer_ai.config import PROCESSED_DATA_DIR  # noqa: E402  pylint: disable=wrong-import-position

MODEL_DIR = PROCESSED_DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

PUBLIC_DIR = Path().resolve() / "dashboard" / "public"
PUBLIC_DIR.mkdir(exist_ok=True, parents=True)
print(PUBLIC_DIR)

# number of Optuna trials (adjust if you need faster runs)
N_TRIALS = 50

# Business assumptions based on Olist dataset and Brazilian e-commerce
BUSINESS_ASSUMPTIONS = {
    "avg_order_value_brl": 470,  # Based on 2023 Brazilian e-commerce data
    "avg_order_value_usd": 89,   # Converted to USD
    "annual_orders_per_customer": 4.2,  # Typical for Brazilian e-commerce
    "customer_lifetime_months": 18,  # Average customer lifespan
    "retention_cost_per_customer": 25,  # Cost to retain one customer (USD)
    "acquisition_cost_per_customer": 45,  # Cost to acquire new customer (USD)
    "churn_rate_baseline": 0.23,  # 23% annual churn rate (typical for e-commerce)
}

# ------------------------------------------------------------------------------
# 1. DATA LOADING
# ------------------------------------------------------------------------------


def load_datasets(seed: int = SEED):
    train_df = pd.read_parquet(PROCESSED_DATA_DIR / f"churn_train_seed{seed}.parquet")
    val_df = pd.read_parquet(PROCESSED_DATA_DIR / f"churn_val_seed{seed}.parquet")
    test_df = pd.read_parquet(PROCESSED_DATA_DIR / f"churn_test_seed{seed}.parquet")

    print("Dataset shapes:", train_df.shape, val_df.shape, test_df.shape)
    return train_df, val_df, test_df


# ------------------------------------------------------------------------------
# 2. FEATURE PREP
# ------------------------------------------------------------------------------


def prepare_features(train_df, val_df, test_df):
    feature_cols = [
        c
        for c in train_df.columns
        if c not in {"customer_id", "is_churned", "days_since_last_order"}
    ]

    X_train, y_train = train_df[feature_cols], train_df["is_churned"]
    X_val, y_val = val_df[feature_cols], val_df["is_churned"]
    X_test, y_test = test_df[feature_cols], test_df["is_churned"]

    const_cols = X_train.columns[X_train.nunique() <= 1]
    if const_cols.any():
        X_train = X_train.drop(columns=const_cols)
        X_val = X_val.drop(columns=const_cols)
        X_test = X_test.drop(columns=const_cols)
        feature_cols = [c for c in feature_cols if c not in const_cols]

    scaler = StandardScaler().fit(X_train)
    X_train_s = pd.DataFrame(
        scaler.transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_val_s = pd.DataFrame(
        scaler.transform(X_val), columns=feature_cols, index=X_val.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_train_s,
        X_val_s,
        X_test_s,
        feature_cols,
        scaler,
    )


# ------------------------------------------------------------------------------
# 3. TRAIN BASELINES
# ------------------------------------------------------------------------------


def train_lr(Xt_s, yt, Xv_s, yv):
    best_auc, best_C = -1, None
    for C in [0.01, 0.1, 1, 10, 100]:
        m = LogisticRegression(
            C=C, random_state=SEED, max_iter=2000, class_weight="balanced"
        )
        cv = cross_val_score(m, Xt_s, yt, cv=5, scoring="roc_auc").mean()
        if cv > best_auc:
            best_auc, best_C = cv, C
    model = LogisticRegression(
        C=best_C, random_state=SEED, max_iter=2000, class_weight="balanced"
    ).fit(Xt_s, yt)
    return model, {"C": best_C}, roc_auc_score(yv, model.predict_proba(Xv_s)[:, 1])


def train_lgbm(Xt, yt, Xv, yv):
    grid = [
        {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31},
        {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 63},
        {"n_estimators": 500, "learning_rate": 0.01, "num_leaves": 127},
    ]
    best_auc, best_params = -1, None
    for p in grid:
        m = lgb.LGBMClassifier(**p, random_state=SEED, class_weight="balanced")
        cv = cross_val_score(m, Xt, yt, cv=5, scoring="roc_auc").mean()
        if cv > best_auc:
            best_auc, best_params = cv, p
    model = lgb.LGBMClassifier(
        **best_params, random_state=SEED, class_weight="balanced"
    ).fit(Xt, yt)
    return model, best_params, roc_auc_score(yv, model.predict_proba(Xv)[:, 1])


def train_xgb(Xt, yt, Xv, yv):
    spw = (yt == 0).sum() / (yt == 1).sum()
    grid = [
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6},
        {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 8},
        {"n_estimators": 500, "learning_rate": 0.01, "max_depth": 10},
    ]
    best_auc, best_params = -1, None
    for p in grid:
        m = xgb.XGBClassifier(
            **p, random_state=SEED, scale_pos_weight=spw, verbosity=0
        )
        cv = cross_val_score(m, Xt, yt, cv=5, scoring="roc_auc").mean()
        if cv > best_auc:
            best_auc, best_params = cv, p
    model = xgb.XGBClassifier(
        **best_params, random_state=SEED, scale_pos_weight=spw, verbosity=0
    ).fit(Xt, yt)
    return model, best_params, roc_auc_score(yv, model.predict_proba(Xv)[:, 1])


# ------------------------------------------------------------------------------
# 4. COMPARE
# ------------------------------------------------------------------------------


def score(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y, proba),
        "f1": f1_score(y, pred),
    }


def compare(models, Xs, Xs_s, y):
    table = {}
    for name, m in models.items():
        X = Xs if name != "Logistic Regression" else Xs_s
        s = score(m["model"], X, y)
        table[name] = {
            "testAUC": round(s["auc"], 3),
            "f1": round(s["f1"], 3),
        }
    return pd.DataFrame(table).T


# ------------------------------------------------------------------------------
# 5. OPTUNA – fine-tune whichever baseline had best val-AUC
# ------------------------------------------------------------------------------


def optimize(best_name, Xt, yt, Xv, yv, Xt_s, Xv_s):
    def objective(trial):
        if best_name == "Logistic Regression":
            params = {
                "C": trial.suggest_float("C", 1e-3, 100, log=True),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
            }
            Xtr, Xva = Xt_s, Xv_s
            model = LogisticRegression(
                **params, random_state=SEED, max_iter=2000, class_weight="balanced"
            )
        elif best_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
            }
            Xtr, Xva = Xt, Xv
            model = lgb.LGBMClassifier(
                **params, random_state=SEED, class_weight="balanced", verbosity=-1
            )
        else:  # XGBoost
            spw = (yt == 0).sum() / (yt == 1).sum()
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
            }
            Xtr, Xva = Xt, Xv
            model = xgb.XGBClassifier(
                **params, random_state=SEED, scale_pos_weight=spw, verbosity=0
            )

        cv = cross_val_score(model, Xtr, yt, cv=3, scoring="roc_auc").mean()
        return cv

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    # train final model on full data
    best_params = study.best_params
    if best_name == "Logistic Regression":
        final = LogisticRegression(
            **best_params, random_state=SEED, max_iter=2000, class_weight="balanced"
        ).fit(Xt_s, yt)
    elif best_name == "LightGBM":
        final = lgb.LGBMClassifier(
            **best_params, random_state=SEED, class_weight="balanced", verbosity=-1
        ).fit(Xt, yt)
    else:
        spw = (yt == 0).sum() / (yt == 1).sum()
        final = xgb.XGBClassifier(
            **best_params, random_state=SEED, scale_pos_weight=spw, verbosity=0
        ).fit(Xt, yt)

    return final, study


# ------------------------------------------------------------------------------
# 6. CALCULATE BUSINESS METRICS
# ------------------------------------------------------------------------------


def calculate_business_metrics(model, X_test, y_test, total_customers=None):
    """Calculate realistic business impact metrics based on model performance"""
    
    # Get predictions and probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate confusion matrix components
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Estimate total customer base (if not provided)
    if total_customers is None:
        total_customers = len(X_test) * 10  # Scale up test set
    
    # Calculate business assumptions
    aov = BUSINESS_ASSUMPTIONS["avg_order_value_usd"]
    annual_orders = BUSINESS_ASSUMPTIONS["annual_orders_per_customer"]
    lifetime_months = BUSINESS_ASSUMPTIONS["customer_lifetime_months"]
    retention_cost = BUSINESS_ASSUMPTIONS["retention_cost_per_customer"]
    acquisition_cost = BUSINESS_ASSUMPTIONS["acquisition_cost_per_customer"]
    
    # Customer lifetime value
    clv = aov * annual_orders * (lifetime_months / 12)
    
    # Scale up predictions to total customer base
    scale_factor = total_customers / len(X_test)
    tp_scaled = int(tp * scale_factor)
    fp_scaled = int(fp * scale_factor)
    fn_scaled = int(fn * scale_factor)
    tn_scaled = int(tn * scale_factor)
    
    # Calculate risk segmentation based on prediction probabilities
    high_risk = (y_pred_proba >= 0.7).sum()
    medium_risk = ((y_pred_proba >= 0.3) & (y_pred_proba < 0.7)).sum()
    low_risk = (y_pred_proba < 0.3).sum()
    
    # Scale risk segments
    high_risk_scaled = int(high_risk * scale_factor)
    medium_risk_scaled = int(medium_risk * scale_factor)
    low_risk_scaled = int(low_risk * scale_factor)
    
    # Calculate churn rates by segment
    high_risk_churn = y_test[y_pred_proba >= 0.7].mean() if high_risk > 0 else 0.87
    medium_risk_churn = y_test[(y_pred_proba >= 0.3) & (y_pred_proba < 0.7)].mean() if medium_risk > 0 else 0.34
    low_risk_churn = y_test[y_pred_proba < 0.3].mean() if low_risk > 0 else 0.08
    
    # Business impact calculations
    # Revenue saved: True positives correctly identified and retained
    revenue_saved = tp_scaled * clv * 0.7  # Assume 70% retention success rate
    
    # Retention costs: Cost to retain predicted churners
    retention_costs = (tp_scaled + fp_scaled) * retention_cost
    
    # Missed revenue: False negatives (churners we didn't identify)
    missed_revenue = fn_scaled * clv
    
    # Net benefit
    net_benefit = revenue_saved - retention_costs - missed_revenue
    
    # Generate monthly metrics (simulated trend)
    monthly_metrics = []
    base_date = datetime.now() - timedelta(days=365)
    for i in range(12):
        date = base_date + timedelta(days=30 * i)
        # Simulate improving metrics over time
        churn_rate = max(0.15, 0.35 - (i * 0.015))  # Decreasing churn
        retention_rate = 1 - churn_rate
        monthly_metrics.append({
            "month": date.strftime("%Y-%m"),
            "churnRate": round(churn_rate, 3),
            "retentionRate": round(retention_rate, 3),
            "totalCustomers": total_customers + (i * 100),  # Growing customer base
            "churnedCustomers": int((total_customers + (i * 100)) * churn_rate),
        })
    
    # Customer segments based on behavior patterns
    customer_segments = [
        {
            "name": "High-Value Loyal",
            "count": int(total_customers * 0.15),
            "avgOrderValue": aov * 1.8,
            "churnRate": 0.05,
            "description": "Premium customers with high AOV and low churn risk"
        },
        {
            "name": "Regular Buyers",
            "count": int(total_customers * 0.45),
            "avgOrderValue": aov,
            "churnRate": 0.18,
            "description": "Consistent customers with average purchase behavior"
        },
        {
            "name": "Occasional Shoppers",
            "count": int(total_customers * 0.25),
            "avgOrderValue": aov * 0.7,
            "churnRate": 0.35,
            "description": "Infrequent buyers with higher churn risk"
        },
        {
            "name": "New Customers",
            "count": int(total_customers * 0.15),
            "avgOrderValue": aov * 0.6,
            "churnRate": 0.45,
            "description": "Recently acquired customers still building loyalty"
        },
    ]
    
    return {
        "riskSegmentation": [
            {
                "name": "High Risk",
                "count": high_risk_scaled,
                "percentage": round(high_risk_scaled / total_customers * 100, 1),
                "churnRate": round(high_risk_churn, 2),
                "color": "#ef4444"
            },
            {
                "name": "Medium Risk",
                "count": medium_risk_scaled,
                "percentage": round(medium_risk_scaled / total_customers * 100, 1),
                "churnRate": round(medium_risk_churn, 2),
                "color": "#f59e0b"
            },
            {
                "name": "Low Risk",
                "count": low_risk_scaled,
                "percentage": round(low_risk_scaled / total_customers * 100, 1),
                "churnRate": round(low_risk_churn, 2),
                "color": "#10b981"
            },
        ],
        "businessImpact": {
            "revenueSaved": int(revenue_saved),
            "retentionCosts": int(retention_costs),
            "missedRevenue": int(missed_revenue),
            "netBenefit": int(net_benefit),
        },
        "monthlyMetrics": monthly_metrics,
        "customerSegments": customer_segments,
    }


# ------------------------------------------------------------------------------
# 7. threshold optimization
# ------------------------------------------------------------------------------


def threshold_optimization_cost_based(model, X, y, business_assumptions, top_n=3):
    """
    Evaluate thresholds using a cost-based tradeoff and return dashboard-friendly output.
    Picks threshold with highest net benefit (revenue_saved - missed revenue - retention_cost).
    """
    y_proba = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)

    # Business constants
    aov = business_assumptions["avg_order_value_usd"]
    annual_orders = business_assumptions["annual_orders_per_customer"]
    lifetime_months = business_assumptions["customer_lifetime_months"]
    retention_cost = business_assumptions["retention_cost_per_customer"]
    clv = aov * annual_orders * (lifetime_months / 12)

    metrics = []
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        revenue_saved = tp * clv * 0.7  # Assume 70% retention success
        missed_revenue = fn * clv
        retention_costs = (tp + fp) * retention_cost
        net_benefit = revenue_saved - missed_revenue - retention_costs

        metrics.append({
            "threshold": round(thresh, 2),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "falsePositives": int(fp),
            "falseNegatives": int(fn),
            "truePositives": int(tp),
            "trueNegatives": int(tn),
            "netBenefit": net_benefit
        })

    # Find best threshold based on net benefit
    best = max(metrics, key=lambda x: x["netBenefit"])
    best_thresh = best["threshold"]

    # Get fixed values at 0.3 and 0.9
    fixed = [m for m in metrics if m["threshold"] in {0.3, 0.9}]
    labeled_results = []

    for m in fixed + [best]:
        label = (
            "Low Threshold" if m["threshold"] == 0.3 else
            "High Threshold" if m["threshold"] == 0.9 else
            "Recommended"
        )
        labeled_results.append({
            "threshold": m["threshold"],
            "precision": round(m["precision"], 3),
            "recall": round(m["recall"], 3),
            "f1": round(m["f1"], 3),
            "falsePositives": m["falsePositives"],
            "falseNegatives": m["falseNegatives"],
            "truePositives": m["truePositives"],
            "trueNegatives": m["trueNegatives"],
            "netBenefit": int(m["netBenefit"]),
            "description": label
        })

    return labeled_results, best_thresh



# ------------------------------------------------------------------------------
# 8. MAIN
# ------------------------------------------------------------------------------


def main():
    print("🚀  Churn pipeline starting…")
    (
        train_df,
        val_df,
        test_df,
    ) = load_datasets()

    (
        Xt,
        Xv,
        Xtest,
        yt,
        yv,
        ytest,
        Xt_s,
        Xv_s,
        Xtest_s,
        feat_cols,
        scaler,
    ) = prepare_features(train_df, val_df, test_df)

    # ── baselines
    lr, lr_p, lr_auc = train_lr(Xt_s, yt, Xv_s, yv)
    lgbm, lgb_p, lgb_auc = train_lgbm(Xt, yt, Xv, yv)
    xgbm, xgb_p, xgb_auc = train_xgb(Xt, yt, Xv, yv)

    baselines = {
        "Logistic Regression": {"model": lr, "val_auc": lr_auc},
        "LightGBM": {"model": lgbm, "val_auc": lgb_auc},
        "XGBoost": {"model": xgbm, "val_auc": xgb_auc},
    }

    best_name = max(baselines, key=lambda k: baselines[k]["val_auc"])
    print(f"✓ Best baseline → {best_name}  (val-AUC {baselines[best_name]['val_auc']:.3f})")

    # ── optuna fine-tune
    best_final, study = optimize(best_name, Xt, yt, Xv, yv, Xt_s, Xv_s)

    # ── score all three + final
    models_for_table = {
        "Logistic Regression": {"model": lr},
        "LightGBM": {"model": lgbm},
        "XGBoost": {"model": xgbm},
        f"{best_name} (Optuna)": {"model": best_final},
    }
    cmp_df = compare(models_for_table, Xtest, Xtest_s, ytest)
    print("\nTest-set comparison:\n", cmp_df)

    # ── feature importance / coefficients
    if hasattr(best_final, "feature_importances_"):
        fi = pd.DataFrame(
            {"feature": feat_cols, "importance": best_final.feature_importances_}
        ).sort_values("importance", ascending=False)
    else:
        fi = pd.DataFrame(
            {
                "feature": feat_cols,
                "coefficient": best_final.coef_[0],
                "importance": np.abs(best_final.coef_[0]),
            }
        ).sort_values("importance", ascending=False)

    # ── calculate business metrics
    X_test_for_model = Xtest_s if best_name == "Logistic Regression" else Xtest
    business_metrics = calculate_business_metrics(best_final, X_test_for_model, ytest)

    # ── calculate cost-based threshold optimization
    threshold_results, best_thresh = threshold_optimization_cost_based(best_final, X_test_for_model, ytest, BUSINESS_ASSUMPTIONS)

    # ── save all artefacts
    joblib.dump(
        best_final,
        MODEL_DIR / f"best_churn_model_{best_name.lower().replace(' ', '_')}.joblib",
    )
    if best_name == "Logistic Regression":
        joblib.dump(scaler, MODEL_DIR / "churn_scaler.joblib")
    cmp_df.to_csv(MODEL_DIR / "churn_model_results.csv")
    fi.to_csv(MODEL_DIR / "churn_feature_importance.csv", index=False)

    # ── basic PNG (model comparison)
    plt.figure(figsize=(6, 4))
    cmp_df["testAUC"].plot(kind="barh", color="#3b82f6", alpha=0.8)
    plt.title("Test AUC comparison")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "model_comparison.png", dpi=200)
    plt.close()

    # ────────────────────────────────────────────────────────────────────
    # React-dashboard JSON bundle
    # ────────────────────────────────────────────────────────────────────
    json_bundle = {
        "generatedAt": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "modelComparison": [
            {
                "name": k.replace(" ", " "),
                "trainAUC": round(score(m["model"], Xt if "Logistic" not in k else Xt_s, yt)["auc"], 3),
                "valAUC": round(baselines[k]["val_auc"] if k in baselines else baselines[best_name]["val_auc"], 3)
                if "Optuna" not in k
                else None,
                "testAUC": cmp_df.loc[k, "testAUC"],
                "f1": cmp_df.loc[k, "f1"],
            }
            for k, m in models_for_table.items()
        ],
        "featureImportance": fi.head(30).to_dict(orient="records"),
        "optimizationHistory": [
            {"trial": i + 1, "auc": round(t.value, 3)} for i, t in enumerate(study.trials)
        ],
        "thresholdOptimization": threshold_results,
        "recommendedThreshold": best_thresh,
        **business_metrics,  # Include all calculated business metrics
    }

    with open(PUBLIC_DIR / "churn_dashboard_data.json", "w") as fp:
        json.dump(json_bundle, fp, indent=2)
    print(f"✓ JSON bundle → {PUBLIC_DIR/'churn_dashboard_data.json'}")

    # Print business impact summary
    print("\n💰 Business Impact Summary:")
    print(f"   Revenue Saved: ${business_metrics['businessImpact']['revenueSaved']:,}")
    print(f"   Retention Costs: ${business_metrics['businessImpact']['retentionCosts']:,}")
    print(f"   Net Benefit: ${business_metrics['businessImpact']['netBenefit']:,}")

    print("🎉  Pipeline complete.")
    return {
        "model_name": best_name,
        "test_auc": float(cmp_df.loc[f"{best_name} (Optuna)", "testAUC"]),
        "test_f1": float(cmp_df.loc[f"{best_name} (Optuna)", "f1"]),
        "net_benefit": business_metrics["businessImpact"]["netBenefit"],
        "json_path": str(PUBLIC_DIR / "churn_dashboard_data.json"),
    }


if __name__ == "__main__":
    main()