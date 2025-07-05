# Customer Segmentation Report  
*K-Means clustering ( **k = 5** ) on Olist Brazilian e-commerce customers*

---

## 1 . Objective  
Create actionable customer segments that marketing, CRM and product teams can target with differentiated campaigns.

---

## 2 . Data & Features  

| Block | Features (all pre-scaled) | Business Rationale |
|-------|---------------------------|--------------------|
| **Transaction** | `frequency`, `avg_order_value`, `tenure_days` | Core RFM-style signals of engagement & value |
| **Payment** | `avg_installments`, `payment_diversity` | Captures credit appetite & payment flexibility |
| **Feedback** | `avg_review_score`, `review_rate` | Post-purchase satisfaction & engagement |
| **Breadth** | `category_diversity` | Propensity to explore catalogue |

*≈ 98 k customers after cleaning; missing values filled with group-level means.*

---

## 3 . Methodology  

| Step | Detail |
|------|--------|
| **Pre-processing** | StandardScaler → μ = 0, σ = 1 |
| **Clustering algorithm** | **K-Means++** (scikit-learn, `n_init=20`, `random_state=42`) |
| **k search** | Evaluated *k* = 2 … 10 |
| **Model-selection metrics** | • **Silhouette Score** ↑ <br>• **Davies-Bouldin Index** ↓ <br>• **Elbow / WSS** (diminishing inertia) |
| **Chosen k** | **5** – balances metric peaks *and* segment interpretability |

---

## 4 . Cluster Snapshot (k = 5)

| Cluster | Size | Key Traits (vs. overall) | Nick-name | Primary Action |
|---------|------|--------------------------|-----------|----------------|
| **0** | **61.9 %** | One-time buyers, low AOV, **high review score** | *First-Timers* | On-boarding drip, next-order coupon |
| **1** | 14.6 % | **Highest AOV ( \$389 )**, long installments | *VIP / High-Value* | Loyalty perks, early access, upsells |
| **2** | 2.3 % | High payment-type diversity | *Multi-Pay Explorers* | Promote flexible-payment promos |
| **3** | 19.1 % | Low review score (1.8), medium spend | *At-Risk* | Service recovery, CSAT survey |
| **4** | 2.1 % | Zero category diversity, lowest spend | *Dormant* | Win-back email, category discovery |

---

## 5 . Financial Impact Estimate  

| Cluster | Avg Order \$ | Avg Freq | \$ / Cust / Yr | Segment Value* |
|---------|--------------|----------|---------------|----------------|
| 0 | \$115 | 1.0 | \$ 115 | \$ 7.1 M |
| 1 | **\$389** | 1.0 | **\$ 389** | **\$ 5.7 M** |
| 2 | \$149 | 1.0 | \$ 149 | \$ 0.3 M |
| 3 | \$135 | 1.0 | \$ 135 | \$ 2.6 M |
| 4 | \$ 93 | 1.0 | \$ 93 | \$ 0.2 M |

\*Assumes current behaviour continues one year and cluster sizes stay constant.

---

## 6 . Recommended Next Steps  

1. **High-Value (Cluster 1)** –  
   *VIP program*, personalised bundles, early-access sales → protect top-line.  
2. **At-Risk (Cluster 3)** –  
   Trigger *CSAT outreach*; monitor NPS before churn occurs.  
3. **Dormant (Cluster 4)** –  
   Reactivation emails featuring new categories & limited-time discounts.  
4. **First-Timers (Cluster 0)** –  
   A/B-test free-shipping threshold to nudge second purchase.  
5. **Model Monitoring** –  
   Re-fit clusters quarterly; track drift in feature distributions.

---

## 7 . Deliverables  

| Artifact | Location |
|----------|----------|
| **`customer_master.parquet`** (with `cluster` column) | `data/processed/` |
| Evaluation plots (`silhouette`, `DB`, `elbow`) | `reports/figures/cluster_selection.png` |
| This report | `reports/customer_segmentation_report.md` |

---

**Bottom line:** targeting **~17 %** of customers (Clusters 1 & 3) with tailored campaigns could influence **≈ \$8 M** in annual revenue, while low-cost nudges to dormant users offer upside with minimal risk.