# 30-day readmission risk: diabetic discharges

**Data:** 30,000 past stays (11.0% readmitted within 30 days) and 2,000 patients being discharged now.
**Model:** TabPFN-3.5 (tabctx), fitted on all 30,000 past stays. `patient_id` was dropped. No other column looked like it leaks the outcome.
**Predictions file:** `patients_current_predictions.csv` (columns `predicted_readmitted_30d`, `p_readmitted_30d=Yes`).

## 1. How accurate is it?

The models were trained on 22,500 past stays and tested on the other 7,500:

| Model | ROC AUC | Log loss | Accuracy |
|---|---|---|---|
| **TabPFN-3.5** (chosen) | **0.697** | **0.321** | 0.890 |
| TabICL-v2 | 0.690 | 0.325 | 0.889 |
| Gradient boosting (scikit-learn baseline) | 0.683 | 0.325 | 0.890 |

- TabPFN-3.5 ranks patients best, but only slightly better than gradient boosting (+0.014 AUC).
- An AUC of about 0.70 is **moderate**. The model is useful for deciding who to contact first. It cannot say for sure who will come back.
- Accuracy tells us little here. Predicting "No" for everyone already scores about 89%.
- The model rarely gives a probability above 50%. Only 3 of the 2,000 current patients are labelled "Yes". **Use the probability to rank patients, not the Yes/No label.**
- The average predicted risk for current patients is 11.2%, close to the past rate of 11.0%.

## 2. Ten highest-risk current patients

| Rank | Patient | P(readmit ≤30d) | Discharge to | Prior inpatient stays | Prior ED visits |
|---|---|---|---|---|---|
| 1 | P025920 | 0.598 | Home | 13 | 4 |
| 2 | P056552 | 0.525 | Short-term hospital | 1 | 0 |
| 3 | P061311 | 0.518 | Home | 11 | 3 |
| 4 | P063665 | 0.493 | Short-term hospital | 3 | 0 |
| 5 | P041248 | 0.491 | Left against advice | 9 | 2 |
| 6 | P036811 | 0.490 | Short-term hospital | 1 | 0 |
| 7 | P032656 | 0.482 | Home | 4 | 0 |
| 8 | P098757 | 0.473 | Short-term hospital | 3 | 2 |
| 9 | P080920 | 0.442 | Rehab facility | 0 | 0 |
| 10 | P051471 | 0.438 | Rehab facility | 2 | 0 |

## 3. What drives readmission

Method: shuffle one column at a time on held-out rows and measure how much AUC falls (permutation importance).

| Factor | AUC drop when shuffled |
|---|---|
| `number_inpatient` (inpatient stays in the past year) | 0.056 |
| `discharge_disposition` (where the patient goes) | 0.039 |
| `number_emergency` (ED visits in the past year) | 0.018 |
| `diag_1` (primary diagnosis group) | 0.014 |
| `num_procedures`, `payer_code`, `medical_specialty`, `time_in_hospital` | 0.004–0.007 each |

- **Past use of the hospital is the strongest signal by far.** Frequent recent inpatient stays and ED visits mark high risk.
- **Where the patient is discharged to comes next.** Transfers to another hospital and leaving against advice stand out in the top-10 list.
- **Diabetes medications and lab results (A1C, glucose) matter very little** to this model. Each moves AUC by 0.003 or less.

## 4. What-if: the three highest-risk patients discharged home

Each row changes one thing (or two) and keeps the rest of the patient's record the same.

| Patient | Profile | Scenario | P(readmit) | Change |
|---|---|---|---|---|
| **P025920** | 20s, 13 inpatient stays and 4 ED visits in the past year, glucose >300, insulin reduced | As is (Home) | 0.598 | — |
| | | Home health service | 0.580 | −0.018 |
| | | Insulin kept steady instead of reduced | 0.596 | −0.002 |
| | | Add metformin (steady) | 0.591 | −0.007 |
| | | Home health + steady insulin | 0.575 | −0.022 |
| **P061311** | 20s, 11 inpatient stays, 3 ED visits, 3 outpatient visits, primary diagnosis diabetes, insulin increased | As is (Home) | 0.518 | — |
| | | Home health service | 0.494 | −0.024 |
| | | Insulin steady instead of increased | 0.518 | 0.000 |
| | | Add metformin (steady) | 0.506 | −0.012 |
| | | Home health + steady insulin | 0.497 | −0.021 |
| **P032656** | 60s, 4 inpatient stays, neoplasm diagnoses, no diabetes medication | As is (Home) | 0.482 | — |
| | | Home health service | 0.379 | **−0.103** |
| | | Start metformin | 0.492 | +0.010 |
| | | Start insulin | 0.501 | +0.019 |
| | | Home health + metformin | 0.380 | −0.102 |

**What this shows:**
- **Home health service** lowered predicted risk for all three patients. The drop was small for the two young patients with very frequent admissions (about 2 points) and large for P032656 (about 10 points).
- **Changing diabetes medication** barely moved the predictions (no more than 1–2 points). For P032656, starting a medication slightly *raised* predicted risk. That most likely reflects who usually gets started on medication (sicker patients), not harm from the drug.
- **P025920 and P061311 stay high-risk (about 50–58%) whatever we change.** Their risk comes from their history of very frequent admissions, which a discharge decision cannot change.

> **Important caveat: these are associations, not proven effects.** The model learned how readmission rates differ between past patients who were discharged home and those discharged with home health service, or between patients on different medications. Those groups also differed in ways the data does not record (who was chosen for home health, how sick they were, their social support, whether they took their medication). A what-if answers: "patients who looked like this but had X were readmitted at rate Y." It does not answer: "if we give this patient X, their risk becomes Y." Only a controlled pilot or trial can show the causal effect.

## 5. Recommended follow-up program

**Home health / transitional-care referral for high-risk patients being discharged home.**

- **Who:** every current patient being discharged to "Home" whose predicted risk is in the top decile. Flag especially those with ≥3 inpatient stays or ≥2 ED visits in the past year.
- **What:** home health service at discharge, a follow-up phone call within 48–72 hours, and medication reconciliation.
- **Super-utilizers** (10+ admissions a year, like P025920 and P061311) need more than this: an intensive case-management pathway. The model expects a home health referral alone to lower their risk only slightly.
- **Expected impact (model-based, not proven):**
  - For the three patients tested, predicted risk dropped by 2 to 10 percentage points, about 5 points on average.
  - Applied to about 120 high-risk home discharges (the top 10% of the 1,231 going home), whose current predicted risk is roughly 20–60%, that is on the order of **3–6 fewer readmissions per 2,000 discharges**.
  - Treat this as an upper-end planning figure. Selection bias probably makes the real effect smaller.
- **How to know if it works:** roll the program out as a randomized or stepped-wedge pilot (some eligible patients referred, some not). Compare actual 30-day readmission rates between the groups. Do not compare model predictions before and after.

## Notes
- The extra file `patients_current_predictions_home.csv` contains the same predictions. It came from a second run ranked among home discharges only. `patients_current_predictions.csv` is the main output.
- Rankings come from a model with AUC of about 0.70. Use them together with clinical judgment.
