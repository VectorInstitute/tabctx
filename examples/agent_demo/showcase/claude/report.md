# 30-day readmission risk: diabetic patients being discharged now

**Data:** 30,000 past stays, of which 11.6% were readmitted within 30 days. There are 2,000 patients being discharged now. `patient_id` was left out of the models. No other column leaks the outcome: everything in the table is known at discharge.

**Predictions file (for scoring):** `patients_current.readmit.predictions.csv`. It has every current patient with `p_readmitted_30d=Yes`. (`home_ranking.csv` is a copy written while ranking the "Home" patients. It holds the same predictions.)

## 1. How well can readmission be predicted?

On 7,500 past stays held out from training:

| Model | ROC AUC | Log loss | Accuracy |
|---|---|---|---|
| **TabPFN-3.5 (chosen)** | **0.677** | **0.336** | 0.884 |
| TabICL-v2 | 0.668 | 0.341 | 0.884 |
| Gradient boosting (scikit-learn baseline) | 0.670 | 0.339 | 0.884 |

- TabPFN-3.5 is the best of the three, but only slightly. It was then fitted on all 30,000 stays.
- **The model is only moderately accurate.** An AUC of 0.68 means it ranks a readmitted patient above one who was not readmitted about two times in three. Accuracy tells us nothing here: predicting "No" for everyone also scores 88.4%. Use the scores to **rank patients for outreach**, not to make yes/no decisions about individuals.
- Across the 2,000 current patients, the mean predicted risk is 10.8%, or about 217 expected readmissions. 162 patients have a risk of 20% or more, 41 have 30% or more, and only 4 are above 50%.

## 2. The 10 highest-risk current patients

| # | patient_id | P(readmit ≤30d) | Discharge to | Prior inpatient stays |
|---|---|---|---|---|
| 1 | P069294 | 0.541 | Short-term hospital | 2 |
| 2 | P062554 | 0.539 | Short-term hospital | 0 |
| 3 | P035606 | 0.532 | Home | 12 |
| 4 | P050933 | 0.513 | Short-term hospital | 1 |
| 5 | P013753 | 0.498 | Short-term hospital | 0 |
| 6 | P093095 | 0.462 | Rehab facility | 0 |
| 7 | P017056 | 0.453 | Rehab facility | 1 |
| 8 | P015152 | 0.435 | Short-term hospital | 0 |
| 9 | P063677 | 0.431 | Home | 7 |
| 10 | P086418 | 0.410 | Home | 7 |

There are two kinds of high-risk patient:

- **Patients transferred to another hospital or to rehab.** Their "readmission" may partly reflect how transfers are recorded.
- **Young patients (age 20–30) whose main diagnosis is diabetes, going home after many earlier admissions.** This is the group a follow-up program can reach.

## 3. What drives readmission

Permutation importance: how much the AUC drops when a column is shuffled (TabPFN-3.5, held-out rows).

| Factor | AUC drop |
|---|---|
| `discharge_disposition` (where the patient goes) | 0.062 |
| `number_inpatient` (inpatient stays in the past year) | 0.052 |
| `diag_1` (primary diagnosis group) | 0.013 |
| `medical_specialty` | 0.009 |
| `number_emergency` (ED visits in the past year) | 0.009 |
| `diabetesMed` (on any diabetes medication) | 0.007 |
| `max_glu_serum` | 0.005 |
| All other columns, including each individual drug, A1C result, length of stay and age | ≤ 0.003 |

- Two factors dominate: **where the patient goes after discharge** and **how often they were admitted in the past year**. Recent ED use and the primary diagnosis add a little.
- Individual diabetes-drug choices matter very little to the model.

## 4. What-if tests: the three highest-risk patients discharged home

All three are aged 20–30, have diabetes as their primary diagnosis, are on insulin, and had no A1C measured.

| Patient | Scenario | P(readmit) | Change |
|---|---|---|---|
| **P035606** (12 prior stays) | As planned (Home, insulin Steady) | 0.532 | — |
| | Home health service | 0.540 | +0.009 |
| | Increase insulin (Up, change = Ch) | 0.527 | −0.005 |
| | Add metformin (Steady, change = Ch) | 0.516 | −0.015 |
| | Home health + increase insulin | 0.536 | +0.004 |
| **P063677** (7 prior stays, 5 ED visits) | As planned (Home, insulin Up) | 0.431 | — |
| | Home health service | 0.434 | +0.003 |
| | Keep insulin Steady (no med change) | 0.454 | +0.023 |
| | Add metformin | 0.428 | −0.003 |
| | Home health + metformin | 0.426 | −0.005 |
| **P086418** (7 prior stays) | As planned (Home, insulin Up) | 0.410 | — |
| | Home health service | 0.425 | +0.015 |
| | Keep insulin Steady (no med change) | 0.431 | +0.021 |
| | Add metformin | 0.404 | −0.006 |
| | Home health + metformin | 0.414 | +0.004 |

What these show:

- **None of the changes moves risk by more than about 2 percentage points.** These patients' risk comes almost entirely from their history of repeated admissions, which nobody can change at discharge.
- **Home health service slightly *raises* predicted risk.** This does not mean home health causes readmissions. In the historical data, patients sent home with home health were sicker, so the model links home health with higher risk.
- Adjusting insulin, rather than leaving it steady, and adding metformin are linked with slightly lower risk. The effects are small, inconsistent and within noise.

> **Caution:** these what-if numbers are **associations the model learned from past records, not proven effects.** Treatment and discharge decisions were not randomized: clinicians chose them based on how sick each patient was, and the data does not fully capture that. The model can say "patients who looked like this and went home with home health were readmitted about this often." It cannot say "sending this patient home with home health will change their risk by X." Only a controlled pilot or trial can measure the real effect of a change.

## 5. Recommended follow-up program

**A transitional-care program for frequent readmitters, targeted by the model's score.**

**Who:** current patients with predicted risk ≥ 20% who are discharged home or home with home health. Across all destinations, 162 of 2,000 patients (8%) are at this level. Give priority to young patients with a diabetes primary diagnosis and several admissions in the past year, such as P035606, P063677 and P086418.

**What:**
- A pharmacist or nurse reviews the insulin plan before discharge.
- An A1C test is done if missing. None of the three patients above had one.
- A phone call within 48 hours.
- A clinic or endocrinology visit within 7 days.
- A direct line to call instead of going to the ED.

The aim is to tackle the main driver the model found, the cycle of repeated admissions and ED visits, rather than to change discharge destination or drug choices. The what-if tests found that changing those barely moves predicted risk.

**Expected impact (an estimate, not a measured result):**

- The 162 targeted patients each have a predicted risk of at least 20%, averaging somewhere around 25–30%. That is roughly 40–45 expected readmissions, about 20% of the ~217 expected across all 2,000 discharges, from 8% of the patients.
- **Assumption:** published transitional-care programs typically reduce readmissions by about 20–30% in relative terms. This figure is not from our data or model.
- On that assumption, the program would avoid roughly **8–13 readmissions per 2,000 discharges**, while reaching only one patient in twelve.
- Run it as a pilot with a randomized or matched comparison group so the real effect is measured. The model's what-if results cannot measure it.
