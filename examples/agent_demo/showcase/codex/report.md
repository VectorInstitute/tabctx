# 30‑Day Readmission Risk — Discharge Cohort (run: Oct 31, 2026)

This report summarizes model performance, current‑patient risk predictions, key drivers, and “what‑if” simulations to inform care‑transition planning.

## 1) Model selection and accuracy

We evaluated two tabular foundation models against a gradient‑boosting baseline on `patients_history.csv` (n=30,000), holding out 25% for testing and dropping the identifier `patient_id`.

- tabpfn‑3.5 — Accuracy 0.884, ROC AUC 0.677, Log loss 0.336
- tabicl‑v2 — Accuracy 0.884, ROC AUC 0.668, Log loss 0.341
- Gradient boosting (baseline) — Accuracy 0.884, ROC AUC 0.670, Log loss 0.339

Chosen model: tabpfn‑3.5 (best ROC AUC).

## 2) Predictions for current discharges

We fitted the chosen model on all history rows and predicted every patient in `patients_current.csv` (n=2,000). Output file (keep for scoring): `predictions_current.csv`.

- Mean predicted probability of readmission (Yes): 0.1083
- Predicted class counts: Yes=4, No=1996

Top 10 highest‑risk patients (by P[Yes]):

1. P069294 — 0.5405
2. P062554 — 0.5388
3. P035606 — 0.5319
4. P050933 — 0.5125
5. P013753 — 0.4978
6. P093095 — 0.4616
7. P017056 — 0.4529
8. P015152 — 0.4353
9. P063677 — 0.4310
10. P086418 — 0.4095

## 3) Main drivers of readmission risk

Permutation importance on held‑out rows (higher = stronger driver):

- Discharge disposition (largest effect)
- Number of prior inpatient visits
- Primary diagnosis group (diag_1)
- Medical specialty on the encounter
- Number of prior emergency visits
- Any diabetes medication (`diabetesMed`)
- Max glucose serum result availability/value
- Race (small), payer code (small), time in hospital (small)

Interpretation: the model’s strongest signals combine where patients are discharged to, recent utilization intensity (inpatient/emergency), clinical category of the index stay, and whether active diabetes meds are on board.

## 4) What‑if simulations for highest‑risk Home discharges

We took the three highest‑risk current patients with `discharge_disposition = Home` and tested practical changes a care team might consider. Numbers below are model‑estimated probabilities; they reflect associations the model learned, not proven causal effects.

Patient P035606 (baseline Home): 0.5318
- Change to Skilled nursing facility: 0.5152  (–0.0166)
- Change to Home health service: 0.5404  (+0.0086)
- Start metformin (set `metformin = Steady`): 0.5237  (–0.0081)

Patient P063677 (baseline Home): 0.4309
- Change to Skilled nursing facility: 0.3870  (–0.0439)
- Change to Home health service: 0.4337  (+0.0028)
- Start metformin (`metformin = Steady`): 0.4284  (–0.0025)

Patient P086418 (baseline Home): 0.4095
- Change to Skilled nursing facility: 0.3974  (–0.0121)
- Change to Home health service: 0.4248  (+0.0153)
- Start metformin (`metformin = Steady`): 0.4038  (–0.0057)

Notes
- For these three patients, “Skilled nursing facility” consistently reduced modeled risk; “Home health service” tended to increase it. This does not mean HH services cause readmissions—only that in the training data, patients going to HH looked more like higher‑risk cases after accounting for other variables.
- Medication adjustments showed small risk shifts; effect sizes vary by clinical context.

## 5) Recommended follow‑up program and expected impact

Program: High‑risk Home‑discharge pathway
- Targeting rule: patients discharged Home with predicted P[readmit] ≥ 0.40.
- Actions:
  - SNF evaluation by case management as the default alternative disposition when clinically appropriate.
  - Diabetes regimen optimization before discharge (e.g., initiate metformin if appropriate, clarify insulin plan).
- Expected impact (association‑based, not causal):
  - Across our three example patients, switching to SNF changed modeled risk by an average –0.024 (–2.4 percentage points). Metformin initiation averaged –0.005 (–0.5 pp).
  - If applied to 30 flagged patients per month with 1/3 placed to SNF and 1/3 receiving med optimization, the model implies ≈ 30 × (0.333×0.024 + 0.333×0.005) ≈ 0.30 fewer 30‑day readmissions per month. Real‑world impact will differ and requires monitoring.

Operationalization & monitoring
- Use `predictions_current.csv` to identify candidates and to track calibration against actual outcomes.
- Build a weekly review: confirm disposition choices, document reasons when SNF not appropriate, and track readmissions for continuous recalibration.
- Equity check: monitor performance and interventions across race/age groups to avoid unintended disparities.

Cautions
- What‑if results reflect patterns in retrospective data; they are not treatment recommendations.
- Always apply clinical judgment and shared decision‑making with patients and families.
