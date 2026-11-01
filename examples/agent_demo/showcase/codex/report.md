# 30‑Day Readmission Risk: Diabetic Patients Being Discharged Today (analysis date: 2026‑10‑31)

This report summarizes model accuracy, the highest‑risk current patients, main drivers of risk, targeted what‑if scenarios for the three highest‑risk Home discharges, and a pragmatic follow‑up program. Predictions are saved to `patients_current.readmit_tabpfn.predictions.csv` for downstream tracking against actual outcomes.

## 1) Model selection and accuracy

We compared two tabular foundation models to a gradient‑boosting baseline on a 7,500‑row holdout from `patients_history.csv` (30,000 rows total), dropping the identifier `patient_id` from features.

- tabpfn‑3.5: Accuracy 0.890; ROC AUC 0.697; log loss 0.3206
- tabicl‑v2: Accuracy 0.889; ROC AUC 0.690; log loss 0.3252
- Gradient boosting: Accuracy 0.890; ROC AUC 0.682; log loss 0.3254

Decision: Use `tabpfn‑3.5` (best ROC AUC = 0.697). AUC is the primary metric here due to class imbalance (3,308 Yes vs 26,692 No in history) and our goal of ranking high‑risk patients.

We then fit `tabpfn‑3.5` on all 30,000 historical rows (dropping `patient_id`) and predicted for all 2,000 current discharges.

## 2) Top 10 highest‑risk current patients (probability of readmission ≤ 30 days)

Sorted by predicted probability `p_readmitted_30d=Yes`:

1. P025920 — 0.5978
2. P056552 — 0.5246
3. P061311 — 0.5181
4. P063665 — 0.4928
5. P041248 — 0.4908
6. P036811 — 0.4903
7. P032656 — 0.4816
8. P098757 — 0.4725
9. P080920 — 0.4421
10. P051471 — 0.4377

File: `patients_current.readmit_tabpfn.predictions.csv` (added columns: `predicted_readmitted_30d`, `p_readmitted_30d=No`, `p_readmitted_30d=Yes`).

## 3) What drives readmission risk (permutation importance)

Top drivers learned from history (higher score means stronger contribution to model discrimination):

- Number of prior inpatient stays (`number_inpatient`) — strongest driver
- Discharge disposition (`discharge_disposition`) — e.g., Home vs facility
- Prior emergency visits (`number_emergency`)
- Primary diagnosis group (`diag_1`)
- Number of procedures this stay (`num_procedures`)
- Payer code (`payer_code`)
- Discharging specialty (`medical_specialty`)
- Length of stay (`time_in_hospital`)
- Gender (`gender`)
- Tertiary diagnosis group (`diag_3`)
- Any diabetes medication this stay (`diabetesMed`)
- Secondary diagnosis group (`diag_2`)

Interpretation: Recent utilization (inpatient/emergency), disposition at discharge, and overall acuity/procedural burden are the dominant signals, with some contribution from diabetes pharmacotherapy being present vs absent.

## 4) What‑if scenarios for top 3 high‑risk Home discharges

These tests modify specific discharge plans/meds for each patient and re‑score risk with the model. They are associations the model has learned, not proven causal effects.

Patient P025920 (baseline 0.5977; predicted Yes)
- Change discharge to Home health service → 0.5797 (−0.0180)
- Set insulin regimen to Steady → 0.5959 (−0.0018)
- Add/continue metformin (Steady) → 0.5910 (−0.0067)

Patient P061311 (baseline 0.5181; predicted Yes)
- Change discharge to Home health service → 0.4938 (−0.0243); flips to No
- Add/continue metformin (Steady) → 0.5059 (−0.0122)
- Set insulin regimen to Steady → 0.5183 (+0.0002; negligible)

Patient P032656 (baseline 0.4816; predicted No but near threshold)
- Change discharge to Home health service → 0.3786 (−0.1030)
- Set insulin regimen to Steady → 0.4756 (−0.0060)
- Add/continue metformin (Steady) → 0.4743 (−0.0073)

Notes
- “Home health service” consistently lowers predicted risk for these three patients, with the largest drop (−10.3 pp) in P032656.
- Diabetes med optimization (metformin/insulin) shows small but favorable risk shifts on average in these cases.
- These are model‑based associations only. They do not establish causal benefit and should be interpreted alongside clinical judgment and resource constraints.

## 5) Recommended follow‑up program and expected impact

Program: Targeted Home‑Health Referral + Med Review for top‑decile Home discharges.

- Who: Current discharges to Home with highest predicted risk (e.g., top 10%).
- Actions: (a) Refer to Home health service at discharge; (b) Pharmacotherapy review to ensure an active, stable diabetes regimen (e.g., metformin where appropriate, insulin plan stabilized/continued rather than “Down” or “No” without a plan).
- Expected impact (associational): In our three test cases, switching to Home health service reduced predicted risk by −1.8 pp, −2.4 pp, and −10.3 pp (avg ≈ −4.8 pp). Medication adjustments yielded smaller average reductions (~−0.7 to −1.2 pp). Actual effects will vary and require monitoring; these are not causal estimates.

## 6) Operational artifacts

- Trained model: tabpfn‑3.5 (fit on all history; `patient_id` dropped)
- Predictions CSV for current patients: `patients_current.readmit_tabpfn.predictions.csv`
- Next steps: (a) Share the top‑risk Home discharges with care‑transition nurses for same‑day action; (b) Track actual 30‑day readmissions to calibrate thresholds; (c) Re‑fit weekly as new outcomes accrue.
