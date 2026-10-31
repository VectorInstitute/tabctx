# Churn Prediction Report (Oct 31, 2026)

Files produced
- Predictions for all current customers: `customers_current.churn_tabicl.predictions.csv` (columns: `predicted_Churn`, `p_Churn=Yes`, `p_Churn=No`).

## 1) Model selection and accuracy
- Data: `customers_history.csv` (5,543 rows; target `Churn`). Dropped id column `customer_id`.
- Compared tabular foundation models vs a standard gradient-boosting baseline on a 75/25 split (1,386 test rows).
- Best by ROC AUC: tabICL v2.
  - tabICL v2: Accuracy 0.799, ROC AUC 0.841, Log loss 0.423.
  - TabPFN 3.5: Accuracy 0.797, ROC AUC 0.841, Log loss 0.421.
  - Gradient boosting: Accuracy 0.777, ROC AUC 0.820, Log loss 0.459.
- Choice: tabICL v2 (slightly better AUC/accuracy and faster inference here).

## 2) Ten highest-risk current customers
(From `p_Churn=Yes`; highest first)
- C04800 — 0.9616
- C03380 — 0.9593
- C02577 — 0.9513
- C06240 — 0.9462
- C00585 — 0.9397
- C02745 — 0.9360
- C04459 — 0.9328
- C05783 — 0.9232
- C02246 — 0.9125
- C05307 — 0.9079

## 3) What drives churn (permutation importance)
Score drop in ROC AUC when shuffling a column (higher = more important):
- Contract — 0.081
- TotalCharges — 0.034
- InternetService — 0.016
- MonthlyCharges — 0.014
- tenure — 0.013
- PaymentMethod — 0.003
- OnlineSecurity — 0.002
- TechSupport — 0.001
- PaperlessBilling — 0.001
- StreamingMovies — 0.001
Interpretation: month-to-month contracts, high monthly charges with low total charges/tenure, fiber internet, electronic check payments, and lack of security/support features are associated with higher churn risk.

## 4) Contract what-if: Top 3 month-to-month customers
Change tested: set `Contract = One year` for each customer.
- C04800: 0.9615 → 0.7962 (−0.1653)
- C03380: 0.9592 → 0.7862 (−0.1730)
- C02577: 0.9512 → 0.7515 (−0.1997)
Average absolute reduction ≈ 0.179 (≈18 percentage points).

## 5) Recommended retention action and expected impact
- Action: Offer a 12‑month price‑lock with a modest monthly discount, switch to automatic bank transfer, and bundle OnlineSecurity/TechSupport for new fiber customers on month‑to‑month with high monthly charges and tenure ≤ 3 months.
- Expected impact: For high‑risk customers who accept, anticipate ~15–20 percentage‑point absolute risk reduction (based on the 3 what‑ifs above).
- Program sizing example: Target the top 300 highest‑risk month‑to‑month customers. If 30% accept, expected avoided churn ≈ 300 × 0.30 × 0.18 ≈ 16–17 customers over the next period.

Notes
- All metrics are out-of-sample on the holdout split. Feature importance is from permutation importance on held‑out rows; near‑zero values indicate weak or no effect in the model.
- The predictions CSV will be used for scoring once actual churn outcomes are known.
