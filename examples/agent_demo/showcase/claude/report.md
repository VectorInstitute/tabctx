# Churn risk report: current customers (2026-10-31)

## 1. How well can churn be predicted?

I held out 25% of `customers_history.csv` (5,543 past customers, 26.5% churned). Each model was fit on the other 4,157 rows and scored on the 1,386 held-out rows. `customer_id` was dropped. No other column looked like it leaked the outcome.

| Model | Accuracy | ROC AUC | Log loss |
|---|---|---|---|
| **tabpfn-3.5 (chosen)** | 0.797 | 0.841 | **0.421** |
| tabicl-v2 | 0.799 | **0.841** | 0.423 |
| Gradient boosting (scikit-learn) | 0.777 | 0.820 | 0.459 |

- Both tabctx models beat gradient boosting: about +0.02 ROC AUC and +2 points of accuracy.
- The two tabctx models tie on ranking (AUC 0.8409 vs 0.8410). I chose **TabPFN-3.5** because it has the lower log loss, so its probabilities are better calibrated, and this report acts on those probabilities.
- What this means in practice: if you pick one customer who churned and one who stayed, the model gives the churner the higher risk about 84% of the time.

The final model was refit on all 5,543 labelled customers.

## 2. Churn risk of current customers

Predictions for all 1,500 current customers are in **`customers_current.churn.predictions.csv`**. It has the columns `predicted_Churn`, `p_Churn=No` and `p_Churn=Yes`.

- 301 customers (20%) are predicted to churn, meaning their probability is above 0.5.
- The mean predicted churn probability is 26.2%. That matches the historical rate, so about 390 departures are expected.

### Top 10 highest-risk customers

| # | customer_id | P(churn) | Contract | Tenure (months) | Internet | Monthly $ | Payment |
|---|---|---|---|---|---|---|---|
| 1 | C03380 | 0.944 | Month-to-month | 1 | Fiber optic | 95.10 | Electronic check |
| 2 | C04800 | 0.943 | Month-to-month | 1 | Fiber optic | 94.00 | Electronic check |
| 3 | C02577 | 0.933 | Month-to-month | 1 | Fiber optic | 94.60 | Electronic check |
| 4 | C06240 | 0.922 | Month-to-month | 1 | Fiber optic | 93.30 | Electronic check |
| 5 | C00585 | 0.920 | Month-to-month | 1 | Fiber optic | 85.70 | Electronic check |
| 6 | C02745 | 0.920 | Month-to-month | 1 | Fiber optic | 86.05 | Electronic check |
| 7 | C05783 | 0.911 | Month-to-month | 1 | Fiber optic | 89.30 | Electronic check |
| 8 | C04459 | 0.898 | Month-to-month | 1 | Fiber optic | 100.25 | Credit card (automatic) |
| 9 | C05307 | 0.889 | Month-to-month | 1 | Fiber optic | 90.10 | Electronic check |
| 10 | C00915 | 0.886 | Month-to-month | 1 | Fiber optic | 73.55 | Electronic check |

All ten fit the same profile: they are in their **first month**, on a **month-to-month** contract, have **fiber-optic** internet, and pay a high monthly bill. Nine of the ten pay by electronic check.

## 3. What drives churn

To measure each column's importance, I shuffled it and measured how much the ROC AUC drops on 500 held-out rows (baseline AUC 0.854):

| Column | AUC drop when shuffled |
|---|---|
| **Contract** | **0.075** |
| InternetService | 0.021 |
| tenure | 0.020 |
| MonthlyCharges | 0.016 |
| TotalCharges | 0.015 |
| PaperlessBilling | 0.004 |
| PaymentMethod | 0.003 |
| everything else | ≈ 0 (not used) |

- **Contract type is by far the strongest driver.** Its effect is more than three times that of any other factor.
- After contract, risk comes from fiber-optic service, short tenure, and high monthly and total charges. Customers who are new and pay a lot are the most likely to leave.
- Demographics (gender, senior citizen, partner, dependents) and the add-on services add almost nothing once those factors are known.
- Caveat: tenure, MonthlyCharges and TotalCharges are correlated, so they partly share importance. Shuffling one of them understates how much that group matters together.

## 4. What if: a one-year contract for the top three month-to-month customers

The three highest-risk customers are all month-to-month. For each one, I changed only `Contract` and left everything else as it is.

| customer_id | Month-to-month (now) | → One year | Change | (→ Two year, for reference) |
|---|---|---|---|---|
| C03380 | 0.944 | 0.757 | **−18.7 pts** | 0.610 |
| C04800 | 0.943 | 0.757 | **−18.6 pts** | 0.637 |
| C02577 | 0.933 | 0.718 | **−21.5 pts** | 0.564 |

- For each of them, a one-year contract lowers predicted risk by about **19–22 percentage points**, which is roughly a 20% relative reduction.
- A two-year contract lowers it by about 30–37 points.
- Even so, all three stay above 50% risk. Their tenure of one month, fiber service and roughly $95 bills still weigh heavily, so a contract alone is not enough for these customers.

Caveat: these are model what-ifs based on patterns in past data, not causal estimates. Customers who chose longer contracts may differ from those who didn't in ways the data doesn't capture. The real effect of *moving* someone to a contract is probably smaller than these numbers, so an A/B test should confirm it.

## 5. Recommended retention action

**Offer new (tenure ≤ 3 months) month-to-month fiber customers a discounted one-year contract, and add an incentive to switch from electronic check to automatic payment.**

- **Who:** the high-risk month-to-month group. 816 current customers are month-to-month, and they make up the entire top-10 list. Start with those whose predicted risk is above 0.7.
- **Expected impact:** the model predicts about 19–22 points less churn risk for each customer who converts. As a rough example: if 100 high-risk customers (average risk about 0.9) are targeted and 30% accept, about 30 × 0.2 ≈ **6 fewer departures per 100 contacted**, before discounting for the causal caveat above.
- **Why a contract offer:** contract type is the strongest and most actionable driver. Tenure and internet type can't be changed directly, but contract terms can be offered.
- **Pair it with:** a first-months price concession for fiber. High monthly charges are the next driver, and the what-if shows these customers stay above 50% risk with a contract change alone.
- **Measure it:** run the offer as a randomized test on a holdout group to measure the true lift.

## Files
- `customers_current.churn.predictions.csv`: the scored predictions for all current customers. This is the file to evaluate.
- `mtm_top3_tmp.csv`: an extra copy of the same predictions, written while filtering to month-to-month customers. It is identical and can be deleted.
