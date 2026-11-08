# Renewal cancellation risk: policies due Nov 2026 – Feb 2027

*For: Head of Retention, Maple Mutual. Prepared 2026-11-05.*

## In short

- **We can predict who cancels.** The model ranks a true canceller above a true
  renewer about 77% of the time (ROC AUC 0.77). That is better than the standard
  gradient-boosting model, which scored 0.72.
- Of the **1,200 policies due**, the model expects about **178 cancellations**
  (14.9%). Last year 16.6% cancelled.
- **The biggest driver is the size of the premium increase.** After that come
  no recent advisor contact, holding only one product with us, short tenure,
  missed payments and paying by cheque or invoice.
- **Recommendation:** an advisor check-in call, with an offer to switch to
  pre-authorized debit (PAD), for the 100 highest-risk policyholders. The model
  expects this to cut their cancellations from about **55 to 31**. That keeps
  about **$35,000 in annual premium**. (This is what the model learned, not a
  proven effect. See the caveats below.)

## 1. What was wrong with the data and how it was fixed

Both exports were cleaned by one script, `clean.py`. It writes
`renewals_2025.csv` (6,000 policies) and `renewals_due.csv` (1,200 policies).

| Problem | Fix |
|---|---|
| A **"TOTAL" row** at the bottom of last year's file (sum of premiums, $12.98M) | Removed: it is not a policy |
| **6 test policies** (`TEST-0001`…`TEST-0006`) in last year's file | Removed |
| **Duplicate rows**: 120 in last year's file, 12 in the due file | Kept one copy of each. The due file still has every one of its 1,200 policies |
| **Outcome written 6 ways**: Renewed / RENEWED / Active / Cancelled / cxl / LAPSED | `cancelled` = **No** for Renewed and Active (still in force), **Yes** for Cancelled, cxl and LAPSED. Result: 994 cancelled, 5,006 renewed |
| **Province written 33 ways** (ON, ont, ONT., Ontario, " on"; PQ, Québec; Alta; Nfld…) | Standardised to 2-letter codes (9 provinces) |
| **Product written 14 ways** (TERM, Term-Life, term life 20, H&D, HEALTH/DENTAL, CI…) | 4 products: Term Life, Whole Life, Health & Dental, Critical Illness |
| **Payment method written 12 ways** (PAD, pre-auth debit, Auto debit; CC, VISA/MC, credit; Cheque, Check, INVOICE…) | 3 methods: Pre-authorized debit, Credit card, Cheque/invoice |
| **Payment frequency written 8 ways** (M, MTHLY, yearly, A…) | Monthly / Annual |
| **Yes/No fields** (advisor, online account, smoker) mixed Y/N, yes/no, TRUE/FALSE, 1/0 | Yes / No |
| **Premiums stored as text**: "$3,040.00", "CAD 2,384", "1247.27" | Converted to plain numbers (dollars) |
| **Premium change** sometimes "4.4%", sometimes "4.4" | Both are percent (the two forms have the same spread of values). Stored as a number, `premium_change_pct` |
| **Impossible ages**: -1, 0, 199, 999 (36 last year, 8 in the due file) | Set to empty (unknown) |
| **"Unknown" written as** n/a, N/A, unknown, "-" in age, advisor contacts and dependents | Set to empty. Nothing was guessed |
| **Dates in 4 formats** (2025-03-27, 2025/08/15, May 23, 2025, 07-Oct-2025) | Parsed every one. Replaced the start and renewal dates with `years_as_customer` at renewal |
| Column names like "Missed Pmts (12m)", "Advisor?" | snake_case: `missed_payments_12m`, `has_advisor`, etc. |

Values still unknown after cleaning were left empty. Last year: age 101,
advisor contacts 225, dependents 268. Due file: 22, 50, 58.

## 2. How accurate the model is

I held out a quarter of last year's policies (1,500), trained on the rest, and
predicted the held-out ones:

| Model | ROC AUC | Accuracy | Log loss |
|---|---|---|---|
| **TabICL v2 (tabular foundation model), chosen** | **0.771** | **84.7%** | **0.377** |
| Gradient boosting (scikit-learn, standard) | 0.722 | 83.8% | 0.411 |

TabICL v2 was the only tabctx model available on our server. It beat gradient
boosting on all three measures. It was then trained on all 6,000 of last
year's policies.

What this means in practice: the model is good at **ranking** who is most at
risk. It is not a crystal ball. Of the 55 due policyholders it rates above 50%,
some will renew, and some cancellations will come from people it rated low.
Use the list to decide who gets attention first.

## 3. The 10 policyholders most likely to cancel

Every due policy's probability is in `renewals_due_predictions.csv` (column
`p_cancelled=Yes`).

| # | Policy ID | Chance of cancelling | Premium increase | Why they stand out |
|---|---|---|---|---|
| 1 | MM-379082 | **95%** | 22.1% | 4 missed payments, pays by cheque, no advisor contact |
| 2 | MM-906997 | **87%** | 20.8% | 2 missed payments, credit card, no advisor contact |
| 3 | MM-623948 | **87%** | 11.0% | 1.4 years with us, 2 missed payments, cheque |
| 4 | MM-611036 | **82%** | 18.5% | credit card, no advisor contact |
| 5 | MM-867409 | **81%** | 21.7% | 2 missed payments, credit card |
| 6 | MM-209749 | **80%** | 3.1% | 3 missed payments, cheque |
| 7 | MM-760981 | **79%** | 20.8% | no advisor contact |
| 8 | MM-367098 | **79%** | 23.6% | 1.6 years with us, credit card |
| 9 | MM-959451 | **77%** | 18.4% | cheque, no advisor contact |
| 10 | MM-757837 | **75%** | 4.2% | under 1 year with us, 3 missed payments |

All ten had **no advisor contact** in the last 12 months and hold **no other
product** with us.

## 4. What drives cancellation

Two sources were used. The first is how much the model's accuracy drops when
each factor is scrambled. The second is last year's actual cancellation rates.
Last year's overall rate was 16.6%.

| Driver | Importance to model | Last year's cancellation rate |
|---|---|---|
| **Premium increase** | by far the largest | ≤5%: **11%** · 5–10%: 18% · 10–15%: 27% · >15%: **36%** |
| **Advisor contacts (12m)** | 2nd | none: **21%** · one or more: **~10%** |
| **Other products held** | 3rd | none: 20% · one: 14% · two or more: **9%** |
| **Years as customer** | 4th | under 2 yrs: 22% · over 10 yrs: 13% |
| **Missed payments (12m)** | 5th | none: 14% · one: 20% · two: 32% · three or more: 50%+ |
| Payment frequency and method | smaller | cheque/invoice: **28%** · credit card: 18% · PAD: **14%** |
| Product | smaller | Term Life 21% · Whole Life 11% |

Age, province, smoker status and number of dependents barely matter.

These are associations. Someone with an advisor contact may be more engaged
to begin with. The call may not be what keeps them.

## 5. What-if tests: the three highest-risk policyholders

I changed one thing at a time and re-scored each person.

| Action | MM-379082 | MM-906997 | MM-623948 |
|---|---|---|---|
| As is | 95% | 87% | 87% |
| Advisor check-in call (+1 contact) | 86% (−9 pts) | 68% (**−19**) | 73% (−13) |
| Move to pre-authorized debit | 91% (−3) | 82% (−5) | 78% (−9) |
| Cap premium increase at 5% (from 22.1 / 20.8 / 11.0%) | 78% (−17) | 56% (**−31**) | 73% (−14) |
| All three together | 55% (−40) | **29% (−58)** | 44% (−43) |

What this says:
- The **premium cap** moves risk the most, when the increase was large.
- The **advisor call** is the best single cheap action. It took 9–19 points
  off for each person.
- **PAD** helps a little on its own.
- Together, the actions add up to a large drop. MM-379082 still has a 55%
  chance even with all three, because of the four missed payments. An
  affordability conversation may matter more than any of these levers.

**Important:** these are **patterns the model learned from last year's data,
not proven effects.** The model sees, for example, that policyholders who had
an advisor contact cancelled less often. It cannot tell whether the contact
caused that. A call that the retention team starts may work better or worse
than the contacts in last year's data. Treat these numbers as a guide to where
to start, and test them (see below).

## 6. Recommended program: "Renewal check-in" for the top 100

**What:** an advisor calls each of the 100 highest-risk due policyholders
before their renewal date. Their predicted risk ranges from 39% to 95%. 96 of
them had no advisor contact last year. On each call the advisor:
1. explains the renewal and the premium change,
2. offers to move them to pre-authorized debit (57 of the 100 aren't on it),
3. may offer a smaller increase if the increase is over 10% and the customer
   is clearly price-sensitive (80 of the 100 face increases over 10%). This
   step is case by case.

**Expected impact (model estimates for these 100 policyholders):**

| Scenario | Expected cancellations | Annual premium at risk |
|---|---|---|
| Do nothing | 55 | $78,000 |
| Call only | 37 | $50,000 |
| **Call + PAD offer (recommended)** | **31** | **$42,000** |
| Call + PAD + cap every increase at 5% | 18 | $24,000 |

- **Call + PAD:** about **24 fewer cancellations** and about **$35,000 of
  annual premium kept**, for roughly 100 phone calls.
- **Adding a 5% cap for everyone** saves about 13 more policies (about
  $18,000). But it gives up roughly $14,500 of premium increases, so in the
  first year it about breaks even. Use it selectively, not across the board.

**Prove it works:** call a random 80 of the 100 and leave 20 alone, then
compare their renewal rates. This is the only way to learn whether the calls
really cause people to stay, rather than relying on the model's patterns. The
result will also sharpen next year's model.

## Files

- `clean.py`: the cleaning script. Re-run with `python3 clean.py`.
- `renewals_2025.csv`, `renewals_due.csv`: the clean tables.
- `renewals_due_predictions.csv`: every due policy with its predicted
  probability of cancelling.
- `scenarios/`: the program what-if inputs and their predictions. These are
  for the impact estimate only; they are not the forecast.
