# Churn Report

## 1. Model accuracy
Held-out test: 25% of `customers_history.csv` (1,386 rows). `customer_id` was dropped.

| Model | Accuracy | ROC AUC | Log loss |
|---|---|---|---|
| tabicl-v2 (chosen) | 0.799 | 0.841 | 0.4225 |
| tabpfn-3.5 | 0.797 | 0.841 | 0.4206 |
| Gradient boosting (baseline) | 0.777 | 0.820 | 0.4588 |

- Both foundation models beat gradient boosting by about 2 points of AUC and 2 points of accuracy. Their log loss is also lower.
- The two are effectively tied. I picked **tabicl-v2** because it has the highest AUC by a hair and is faster. TabPFN's log loss is marginally better.
- Churn is about 27% of history, so a model that always says "No" would score about 73% accuracy. The model's lift is real but modest. Use it to rank customers rather than to make hard yes/no calls.

## 2. Predictions
All 1,500 current customers were scored with tabicl-v2. The file is `customers_current.churn_icl.predictions.csv` (columns `predicted_Churn`, `p_Churn=No`, `p_Churn=Yes`).

- 308 customers are predicted to churn (20.5%).
- The mean churn probability is 0.256.

### 10 highest-risk customers
| Rank | customer_id | P(churn) |
|---|---|---|
| 1 | C04800 | 0.9615 |
| 2 | C03380 | 0.9594 |
| 3 | C02577 | 0.9511 |
| 4 | C06240 | 0.9460 |
| 5 | C00585 | 0.9393 |
| 6 | C02745 | 0.9355 |
| 7 | C04459 | 0.9327 |
| 8 | C05783 | 0.9229 |
| 9 | C02246 | 0.9125 |
| 10 | C05307 | 0.9074 |

All ten are on a month-to-month contract with fiber-optic internet. All have tenure of 1 month and pay about $86–$102 a month.

## 3. Churn drivers
These come from permutation importance for tabpfn-3.5, measured as the drop in AUC when a column is shuffled. I did not run it for tabicl-v2.

| Factor | AUC drop |
|---|---|
| **Contract** | 0.075 (by far the largest) |
| InternetService | 0.021 |
| tenure | 0.020 |
| MonthlyCharges | 0.016 |
| TotalCharges | 0.015 |
| PaperlessBilling | 0.004 |
| PaymentMethod | 0.003 |

The other columns (gender, partner, dependents, security and support add-ons, and so on) contribute about nothing.

- The pattern is that churners are new customers on flexible contracts, on fiber optic, and paying high monthly charges.
- The importance scores show which columns the model uses. They do not show causation.

## 4. What-if: month-to-month → one-year contract
I took the three highest-risk month-to-month customers. They are also the top three overall. I changed only `Contract` and left every other field as it was.

| customer_id | Current P(churn) | With one-year contract | Change |
|---|---|---|---|
| C04800 | 0.9615 | 0.7953 | −0.166 (−17.3% relative) |
| C03380 | 0.9592 | 0.7858 | −0.173 (−18.1%) |
| C02577 | 0.9512 | 0.7517 | −0.200 (−21.0%) |

- A one-year contract lowers predicted risk by roughly 17–20 percentage points for each of them.
- All three **remain high risk** (above 0.75) and are still predicted to churn. A contract change alone is not enough for these customers.
- This is a model-based what-if on historical associations. Customers who choose a longer contract may differ from those who are moved onto one, so the true effect could be smaller.

## 5. Recommended action
**Offer new month-to-month fiber customers, in their first months, a discounted one-year contract. Pair it with an onboarding check-in.**

- **Target:** month-to-month customers with tenure under about 3 months, starting with the ~300 predicted-to-churn customers. Prioritize by `p_Churn=Yes`.
- **Expected impact:** for the top-risk customers, the one-year switch lowers predicted churn probability by about 17–20 points. If 50% accept, that is roughly 8–10 points of risk removed per targeted customer. Among the top 10, that is about one customer retained for every 10–12 offers.
- Because these customers are still above 75% risk after the switch, add a second lever. For example, a first-year fiber price lock or a service-quality call.
- Run it as an A/B test, holding out a random group, to measure the real effect before rolling out widely.
