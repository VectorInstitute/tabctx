# 30-day readmission risk: diabetic patients being discharged

## 1. How accurate is the model?

Held out 25% of `patients_history.csv` (7,500 of 30,000 stays). `patient_id` was dropped.

| Model | ROC AUC | Accuracy | Log loss |
|---|---|---|---|
| **tabpfn-3.5** (chosen) | **0.677** | 0.884 | 0.336 |
| tabicl-v2 | 0.668 | 0.884 | 0.341 |
| sklearn gradient boosting (baseline) | 0.670 | 0.884 | 0.339 |

- TabPFN is the best tabctx model. It beats gradient boosting by a small margin (+0.007 AUC, slightly better log loss). That gap is small, so I would call it roughly a tie with a slight edge.
- **Readmission is only weakly predictable from this data.** AUC 0.68 is modest.
- The 0.884 accuracy is almost meaningless. About 88% of stays are not readmitted (3,478 of 30,000 were), so always answering "No" scores the same.
- Use the output as a **ranking and prioritisation** tool, not a yes/no call. The model flags only 4 of 2,000 current patients as "Yes" at a 50% cutoff.
- The highest predicted probability is 54%. Mean predicted risk for the current cohort is 10.8%, against 11.6% in the history.

## 2. Predictions

All 2,000 current patients were scored with the TabPFN model fitted on all 30,000 historical stays.
File: `patients_current.pfn.predictions.csv`. It has `predicted_readmitted_30d`, `p_readmitted_30d=No` and `p_readmitted_30d=Yes`.
`home_top_scratch.csv` is a scratch by-product of the what-if selection step. Ignore it.

### 10 highest-risk patients

| # | patient_id | P(readmit 30d) | Discharge disposition | Prior inpatient stays |
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

## 3. What drives readmission

Permutation importance (drop in held-out AUC when a column is shuffled):

1. **`discharge_disposition`** (0.062). This is the largest driver. Transfers to another hospital or a rehab facility rank high. Most of the top 10 above are in these groups.
2. **`number_inpatient`** (0.052), the count of prior inpatient stays. More prior stays means higher risk.
3. `diag_1`, the primary diagnosis (0.013).
4. `medical_specialty` (0.009), `number_emergency` (0.009), `diabetesMed` (0.007), `max_glu_serum` (0.005).
5. Everything else is at or near zero, including the individual diabetes drugs and `change`. Age, length of stay and lab counts add almost nothing.

In short, risk is driven mainly by where the patient goes next and by how often they have been admitted before. It is driven very little by the diabetes regimen.

## 4. What-if tests (highest-risk patients discharged Home)

The three are P035606, P063677 and P086418. All are aged 20-30, have a primary diagnosis of Diabetes, and have 7 to 12 prior inpatient stays.
Each row changes one thing and holds the rest fixed.

| Scenario | P035606 | P063677 | P086418 |
|---|---|---|---|
| Baseline (Home) | 0.532 | 0.431 | 0.410 |
| Home health service | 0.540 (+0.9 pt) | 0.434 (+0.3 pt) | 0.425 (+1.5 pt) |
| Insulin held "Steady", no med change (P035606: insulin Up, changed) | 0.527 (-0.5 pt) | 0.454 (+2.3 pt) | 0.431 (+2.1 pt) |
| Insulin "Down" (P035606: insulin No, diabetesMed No, changed) | 0.510 (-2.2 pt) | 0.439 (+0.7 pt) | 0.417 (+0.7 pt) |
| Home health + insulin steady/no change (P035606: home health + insulin up) | 0.536 (+0.4 pt) | 0.455 (+2.4 pt) | 0.441 (+3.2 pt) |

**Reading these results:**
- Almost every change moves predicted risk by 2 points or less, and most of them move it **up**. The model finds no lever that clearly helps.
- The changes are small and inconsistent in direction across the three patients. They are within the noise of a model with AUC 0.68.
- **These are associations the model learned, not causal effects.** The model learned from past data that Home health service patients were readmitted at similar or higher rates. That is most likely because sicker patients are the ones sent to home health. Swapping that label on a patient does not make the patient healthier, and it says nothing about what home health would do.
- The same applies to medication changes. In the history, sicker patients get their regimens changed. The model also did not see the intervention itself, such as nurse calls, only the disposition label.
- These three patients are unusual: aged 20-30 with 7 to 12 prior inpatient stays. The history has very few comparable patients, so the model is extrapolating and the numbers for them are less reliable.
- Treat the what-ifs as hypotheses to test, not as expected savings.

## 5. Recommended follow-up program

**Transitional-care follow-up for patients with frequent prior inpatient stays or a transfer-type discharge.** It would be a nurse call within 48 hours, a follow-up visit within 7 days, and medication reconciliation. Patients would be enrolled by ranking on the model's probability, with `number_inpatient` and `discharge_disposition` as the practical screening rules.

- **Why this program:** the two strongest, most stable signals are prior inpatient use and discharge destination. The what-ifs do not support changing diabetes drugs or switching disposition labels.
- **Expected impact (an assumption, not a result from this data):** the model cannot estimate the effect of the program. Impact per patient enrolled is the patient's baseline risk times the program's relative effect. Even the highest-risk patients here sit at about 40-54%. If a program cut readmissions by a relative 10-20%, which is an assumption, then enrolling the top 10 (average risk about 0.48) would avoid roughly 0.5 to 1 readmission among them. Enrolling 100 patients at an average 25% risk would avoid roughly 2.5 to 5.
- **Limits on targeting:** with AUC 0.68, targeting is only moderately better than random. Expect many enrolled patients not to be readmitted, and many readmissions among patients who were not enrolled.
- **Measure it properly:** randomise or stagger enrolment among eligible high-risk patients, for example by week or unit. Compare readmission rates, then refit the model on the new data.
