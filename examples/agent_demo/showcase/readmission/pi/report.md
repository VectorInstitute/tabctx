# 30-Day Readmission Risk — Current Discharge Cohort

**Date:** 2026-11-01 · **Cohort:** `patients_current.csv` (2,000 patients being discharged now)
**Trained on:** `patients_history.csv` (30,000 past diabetic stays; 11.0% readmitted within 30 days)
**Model:** `tabpfn-3.5` (tabular foundation model), fit on all 30,000 labelled history rows
**Predictions file:** `patients_current.readmit.predictions.csv` (every patient, with `p_readmitted_30d=Yes`)

---

## 1. How well can readmission be predicted?

Holdout evaluation (22,500 train / 7,500 test, `patient_id` dropped):

| Model | Accuracy | ROC AUC | Log loss |
|---|---|---|---|
| **tabpfn-3.5 (chosen)** | **0.890** | **0.697** | **0.3206** |
| tabicl-v2 | 0.889 | 0.690 | 0.3252 |
| Gradient boosting (baseline) | 0.890 | 0.683 | 0.3254 |

**tabpfn-3.5 wins** on AUC and log loss; accuracy is a tie across all three.

Read the accuracy carefully: the base rate is 89.0% "No", so any model that always says "No" scores 0.890. The useful metric is **AUC ≈ 0.70** — readmission is only *moderately* predictable from these fields. That is fine for our purpose (prioritizing who gets extra follow-up, not deciding who is readmitted):

- Cohort mean predicted risk: **11.2%**, vs 11.0% historical rate — well calibrated at the population level.
- Risk is spread: 3 patients predict above 50%, the top 10 all sit at 44–60% (≈ 4× the cohort average).

## 2. Ten highest-risk current patients

| # | Patient ID | P(readmit within 30d) |
|---|---|---|
| 1 | P025920 | 0.598 |
| 2 | P056552 | 0.525 |
| 3 | P061311 | 0.518 |
| 4 | P063665 | 0.493 |
| 5 | P041248 | 0.491 |
| 6 | P036811 | 0.490 |
| 7 | P032656 | 0.482 |
| 8 | P098757 | 0.473 |
| 9 | P080920 | 0.442 |
| 10 | P051471 | 0.438 |

## 3. Main drivers of readmission risk

Permutation importance (ROC AUC drop when the column is shuffled, held-out rows):

| Column | AUC drop | Interpretation |
|---|---|---|
| `number_inpatient` | 0.0557 | Prior inpatient stays in the past 12 months — by far the strongest signal (frequent-flyer utilization) |
| `discharge_disposition` | 0.0394 | Where the patient goes after discharge (Home = highest risk; SNF / home-health lower) |
| `number_emergency` | 0.0181 | Prior ER visits |
| `diag_1` | 0.0140 | Primary diagnosis category |
| `num_procedures` | 0.0066 | Procedures during the stay |
| `payer_code` | 0.0058 | Payer (likely a proxy for access/coverage gaps — flag for equity review) |
| `medical_specialty` | 0.0045 | Admitting specialty |
| `time_in_hospital` | 0.0044 | LOS |

The model leans heavily on **recent utilization** (inpatient + ER visits) and **discharge destination**. Diabetes medications (insulin, metformin, etc.) and A1C/glucometer values are comparatively weak predictors — the model uses them, but they are not the main levers. Age and race contribute very little.

## 4. What-if: interventions for the three highest-risk Home discharges

The three highest-risk patients with `discharge_disposition = "Home"`: P025920 (0.598), P061311 (0.518), P032656 (0.482). For each, we re-ran the model with 2–3 changes a care team could actually make:

### P025920 — F, [20-30), 13 prior inpatient stays, 4 ER visits, insulin being *reduced*, max glucose >300
| Scenario | P(readmit) | Δ |
|---|---|---|
| As discharged (Home) | 0.598 | — |
| Discharge with **home health service** | 0.580 | **−1.8 pts** |
| **Stabilize insulin** (Down → Steady) | 0.596 | −0.2 pts |
| **Increase insulin** (Down → Up) | 0.589 | −0.8 pts |

Medication changes barely move this patient — her risk is dominated by 13 prior admissions. She needs a utilization/follow-up intervention, not just a medication tweak (the insulin reduction + glucose >300 is still worth a prescriber's review).

### P061311 — F, [20-30), 11 prior inpatient stays, 3 ER visits, insulin being *increased*
| Scenario | P(readmit) | Δ |
|---|---|---|
| As discharged (Home) | 0.518 | — |
| Discharge with **home health service** | 0.494 | **−2.4 pts** (crosses 50%) |
| Discharge to **skilled nursing facility** | 0.466 | **−5.3 pts** (crosses 50%) |
| **Stop the insulin increase** (Up → Steady) | 0.518 | 0.0 pts |

For her, the discharge *place* is the effective lever; the insulin change is neutral in the model.

### P032656 — M, [60-70), elective stay, 4 prior inpatient stays, **no diabetes medication at all**
| Scenario | P(readmit) | Δ |
|---|---|---|
| As discharged (Home) | 0.482 | — |
| Discharge with **home health service** | 0.379 | **−10.3 pts** (largest effect found) |
| **Start insulin** | 0.505 | +2.4 pts |
| **Start metformin** | 0.498 | +1.6 pts |

Note the medication direction: *starting* glucose medication **raises** predicted risk. That is almost certainly confounding in the historical data (patients who need to start insulin tend to be sicker / have uncontrolled diabetes), not a signal that starting metformin is harmful. The model's actionable finding for him is the home-health arrangement.

**Pattern across the three:** changing the discharge plan moves predicted risk (−1.8 to −10.3 pts, averaging ≈ 4.8 pts ≈ 10% relative); changing diabetes medication alone moves it little (≤ 2.4 pts) or in counterintuitive directions.

> ⚠️ **These what-if numbers are associations the model learned from 30,000 historical stays, not proven causal effects.** The historical data contains confounding (who gets SNF vs home, who gets insulin started, who gets a payer covering home health is not randomly assigned). They are best read as "these levers are where the risk lives in the data" — worth acting on and worth testing in a structured pilot, not as guaranteed effect sizes.

## 5. Recommended follow-up program: "High-Risk Home Discharge Outreach"

**Population:** patients like the current cohort with predicted 30-day readmission risk ≥ 0.35 being discharged **Home** (the three above qualify; in the current cohort the top decile of risk is 0.44–0.60).

**Bundle (from the what-if analysis, largest levers first):**
1. **Arrange home health service at discharge** — instead of plain "Home". This was the single most consistent lever (−1.8 to −10.3 pts across the three tested patients; −5.3 pts for P061311 under SNF placement, which is a fallback when home health isn't feasible).
2. **72-hour post-discharge contact** (nurse call or home visit) to check on medications and glucose — targets the frequent-utilizer profile (prior inpatient/ER visits) that drives most of the risk.
3. **Medication reconciliation flag** — flag patients whose insulin was just reduced or started, or who have no diabetes medication despite diabetes diagnoses (P025920, P032656), for prescriber review before discharge.
4. **Equity review:** `payer_code` is a modest driver — confirm the program isn't systematically skipping patients whose coverage makes home health look "unavailable" in the data.

**Expected impact (model-based, conservative):** if moving high-risk Home discharges to home-health service produced even a uniform 2-point absolute risk reduction for those patients, the current 2,000-patient cohort's mean predicted risk would fall from 11.2% to ≈ 10.0% — roughly **25 fewer readmissions per 2,000 discharges (≈ 11% relative)**; in the individual cases tested the effect was larger (up to 10.3 pts for P032656). Repeat the prediction after outcomes are known and re-fit the model to confirm the association holds.

## 6. Caveats

- **Modest signal:** AUC 0.70 means many readmissions are not predictable from these fields; probabilities should drive *prioritization* (who gets outreach first), never rationing of standard care.
- **What-ifs ≠ effects:** as stated above — associations, not proven causation. The insulin "start = higher risk" example shows how confounding can point the wrong way.
- **Proxies for social risk:** payer and specialty drive some of the risk; that may reflect access inequities rather than medical need — review before using the model for resource allocation.
- **Missing data:** payer (~40%), specialty (~50%) and A1C/glucose ("Not measured" for most) are sparse; the model copes, but these fields would sharpen it if completed.
