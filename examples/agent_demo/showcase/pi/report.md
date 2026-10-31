# 30-day readmission risk: current discharges

## 1. How accurate is the model?

I held out 25% of `patients_history.csv` (7,500 of 30,000 stays) and scored each model on it. `patient_id` was dropped. The readmission rate is 11%.

| Model | ROC AUC | Log loss | Accuracy |
|---|---|---|---|
| **tabpfn-3.5** (chosen) | **0.697** | **0.3206** | 0.890 |
| tabicl-v2 | 0.690 | 0.3252 | 0.889 |
| sklearn gradient boosting (baseline) | 0.683 | 0.3254 | 0.890 |

- **Choice:** TabPFN-3.5 is best on AUC and log loss. Its lead over gradient boosting is small (+0.015 AUC), but it is consistent across both metrics.
- **Ranking ability is modest.** An AUC of about 0.70 means the model ranks a readmitted patient above a non-readmitted one about 70% of the time. That is useful for prioritising, but it is far from certain for any one patient.
- **Ignore the accuracy figure.** Always answering "No" would score about 89%. The model flags only 3 of 2,000 current patients as "Yes" at the 50% threshold. Use the probabilities and rank patients by them.
- **Probabilities are low overall.** The mean predicted risk is 11.2%. Only 3 patients are above 50%, and the highest is 59.8%.

The final model, `tabpfn_readmit`, was refit on all 30,000 history rows.

## 2. Predictions

Predictions for all 2,000 current patients are in `predictions.csv`. The columns are `predicted_readmitted_30d`, `p_readmitted_30d=No` and `p_readmitted_30d=Yes`.

### 10 highest-risk patients

| Rank | patient_id | P(readmit 30d) | Discharge | Prior inpatient stays |
|---|---|---|---|---|
| 1 | P025920 | 0.598 | Home | 13 |
| 2 | P056552 | 0.525 | Short-term hospital | 1 |
| 3 | P061311 | 0.518 | Home | 11 |
| 4 | P063665 | 0.493 | Short-term hospital | 3 |
| 5 | P041248 | 0.491 | Left against advice | 9 |
| 6 | P036811 | 0.490 | Short-term hospital | 1 |
| 7 | P032656 | 0.482 | Home | 4 |
| 8 | P098757 | 0.473 | Short-term hospital | 3 |
| 9 | P080920 | 0.442 | Rehab facility | 0 |
| 10 | P051471 | 0.438 | Rehab facility | 2 |

## 3. Main drivers

These come from permutation importance. The score is the drop in held-out AUC when a column is shuffled. The test used only 500 held-out rows, so small values are noisy.

| Factor | AUC drop |
|---|---|
| `number_inpatient` (prior inpatient stays) | 0.056 |
| `discharge_disposition` | 0.039 |
| `number_emergency` (prior ER visits) | 0.018 |
| `diag_1` (primary diagnosis group) | 0.014 |
| `num_procedures` | 0.007 |
| `payer_code`, `medical_specialty`, `time_in_hospital` | about 0.004–0.006 |

- **Prior utilisation dominates.** Patients with many earlier inpatient stays or ER visits are the most likely to come back.
- **Discharge destination is the second-largest factor.** It is also the only large driver a care team can change at discharge.
- **Diabetes medications and lab results add little.** Individual drugs, `A1Cresult`, `change` and `diabetesMed` are all at or near zero, meaning the model barely uses them.
- **Limit of this method.** Permutation importance shows what the model relies on. It does not show what causes readmission.

## 4. What-if analysis (top 3 patients discharged Home)

The three highest-risk patients with `discharge_disposition` = Home are P025920, P061311 and P032656. I changed one thing at a time and re-predicted. Values are P(readmit) with the change from baseline in percentage points.

| Scenario | P025920 | P061311 | P032656 |
|---|---|---|---|
| Baseline (Home) | 59.8% | 51.8% | 48.2% |
| Home health service | 58.0% (−1.8) | 49.4% (−2.4) | 37.9% (−10.3) |
| Skilled nursing facility | 54.4% (−5.4) | 46.6% (−5.2) | 38.3% (−9.9) |
| Medication change* | 60.4% (+0.6) | 51.8% (0.0) | 50.1% (+1.9) |
| A1C recorded as "Norm" | 59.5% (−0.3) | 51.4% (−0.4) | 45.8% (−2.4) |

\*The medication change differs by patient:
- P025920: insulin "Down" → "Steady", `change` = "No".
- P061311: insulin "Up" → "Steady", `change` = "No".
- P032656: starting insulin "Steady" (`change` = Ch, `diabetesMed` = Yes) in a patient with no diabetes medication now.

**Findings**
- **Discharge destination moves risk the most.** Moving to home health or a skilled nursing facility lowers predicted risk for all three patients. The effect is small for the two patients with many prior stays (about 2–5 points). It is larger for P032656 (about 10 points).
- **Medication and A1C changes do nothing useful.** The effects are within about ±2 points, which is probably noise given how little the model uses these columns. I would not build a program around them.
- **Even after the best change, none of the three drops below about 38%.** Their risk comes mainly from their history, which no discharge-day action alters.

**These are associations the model learned, not proven effects.**
- The model learned that, in past data, patients discharged to home health or a skilled nursing facility were readmitted at different rates from similar patients sent home.
- Those placements were not randomly assigned. Clinicians chose them based on factors the data may not capture, such as frailty, home support and clinical judgment.
- Changing a row's destination in the model does not show that sending a real patient there would lower their risk.
- Skilled nursing facility and home health also differ in cost, capacity and eligibility.
- The medication what-ifs are weaker still, since the model barely uses those columns.
- A pilot or randomised comparison is needed before claiming any benefit.

## 5. Recommended follow-up program

**Program: a transition-of-care bundle for high-prior-utilisation patients.**
- **Who:** patients in the top 10% by predicted risk (about 200 of 2,000). They are mostly patients with several prior inpatient or ER encounters.
- **What:** a home health referral where eligible, a nurse call within 48 hours, and a primary-care visit within 7 days.
- **Where it should start:** with patients discharged Home who have high risk. For them the model shows a possible benefit from a home health referral.

**Expected impact (illustrative only):**
- The what-if results show a drop in predicted risk of about 2–10 points per patient from a home health referral. The two highest-risk patients are at the low end, about 2 points.
- The expected drop therefore depends on who is enrolled. Take roughly 200 patients, an average drop of about 2–5 points, and a modest average of about 4 points as the middle case. That prevents about 8 readmissions, with a plausible range of 4–20.
- This is an association-based estimate with wide uncertainty. Risk reductions may be smaller in practice.
- **Pilot design:** randomise eligible high-risk patients to the bundle or usual care, and track actual 30-day readmissions. This pilot is needed before any expected impact is relied on.

## Caveats
- AUC is about 0.70, so individual predictions are uncertain.
- The model was trained on past stays, and the current patients may differ slightly. For example, some patients have rare categories, such as an age of 20–30 with 13 prior inpatient stays. Those predictions deserve a manual check.
