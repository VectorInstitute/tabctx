# Group plan renewals: who is likely to leave, and what might keep them

For: Head of Group Retention · Prepared 2026-11-18 · Covers the 1,200 plans renewing between mid-November 2026 and mid-February 2027

## In brief

- We expect about **131 of the 1,200 due plans to leave** (11%). That puts about **$17.9M of annual premium at risk** (each plan's premium weighted by its chance of leaving).
- The plans most at risk are **small employers facing a large renewal increase (over 10%) who had no account review last year**. Service issues and late payments raise the risk further.
- **Recommended program:** before renewal, hold an account review with the 200 highest-risk due plans that had none last year. The model suggests this could cut expected departures in that group from about 59 to about 27, keeping roughly **$3.3M of premium**. That is a pattern in past data, not a proven effect, so run it as a test (see below).

## 1. What was wrong with the data, and how it was fixed

Both exports had the same problems. `clean.py` fixes them and writes `renewals_2025.csv` (6,000 plans) and `renewals_due.csv` (1,200 plans, every due plan kept).

| Problem | What we did |
|---|---|
| Rows that are not plans in the 2025 export: 6 test plans (`TEST-0001` to `TEST-0006`) and a `TOTAL` line at the bottom ($1.27B) | Removed |
| Exact duplicate rows: 120 in the 2025 export, 12 in the due export | Kept one copy of each. After that, every plan ID appears once. |
| Renewal outcome written 6 ways (`Renewed`, `RENEWED`, `In force`, `Terminated`, `TERM`, `moved carrier`) | Made a `terminated` column: **Yes** = Terminated / TERM / moved carrier, **No** = Renewed / RENEWED / In force. "In force" plans look the same as renewed plans (same renewal increases and service issues), so we counted them as staying. |
| Provinces written 33 ways (`ON`, `ont`, ` on`, `ONT.`, `Ontario`, `PQ`, `Québec`, `Alta`, `Nfld` …) | Changed to the standard two-letter codes (10 provinces) |
| The same category under several names in industry, plan type, funding, channel and billing (e.g. `MFG`/`Mfg.`, `HDL`/`H/D/Life`, `INSURED`/`Fully Insured`, `A.S.O.`/`ASO`, `BRKR`/`broker`, `Check`/`Cheque`) | Mapped to one name each: 7 industries, 3 plan types, 3 funding types, 3 channels (`In-house` counted as Direct), 3 billing methods (PAD, EFT, Cheque/invoice) |
| Yes/no columns written as `Y`, `yes`, `TRUE`, `1`, `N`, `no`, `FALSE`, `0` | Changed to Yes / No |
| Premium written as `62770.46`, `$74,848.88` or `CAD 9,676` | Converted to plain numbers in dollars |
| Renewal increase sometimes `8.1%`, sometimes `8.1` (same scale) | Converted to a percent number |
| Loss ratio sometimes a percent (`63.2%`), sometimes a fraction (`0.856`) | Converted all to percent (0.856 → 85.6) |
| Dates in 4 formats (`2025-01-07`, `2025/01/07`, `Jan 07, 2025`, `07-Jan-2025`) | Read all four. Replaced the two dates with `years_with_us` (renewal date minus plan start date) |
| Placeholder values for "unknown": `n/a`, `N/A`, `unknown`, `-` in lives covered, account reviews and member app use. Impossible lives counts: `-1`, `0`, `99999`, `999999` (51 plans across both files) | Left empty (missing). We did not guess these values. |

The script stops with an error if it finds a category value it does not know, so a new spelling will not get through unnoticed.

## 2. How accurate the model is

We kept 1,500 of last year's plans aside, trained on the other 4,500, and checked the predictions against what really happened:

| Model | Ranking quality (ROC AUC) | Accuracy |
|---|---|---|
| **TabICL v2 (chosen)** | **0.765** | 87.5% |
| TabPFN 3.5 | 0.765 | 87.3% |
| Standard gradient boosting | 0.733 | 86.9% |

- Both foundation models beat the standard gradient-boosting model. They are tied with each other, and we used TabICL v2.
- **Do not read too much into accuracy.** Only 12.7% of plans left last year, so a rule that says "everyone renews" is already 87% accurate.
- The better measure is ranking quality. An AUC of 0.77 means that if you pick one plan that left and one that stayed, the model gives the leaver the higher risk score about 3 times in 4. That is good enough to decide who to call first. It is not good enough to be sure about any one employer.
- The 17 plans above 50% and the 86 plans above 30% are the realistic list to act on.

The model was then trained on all 6,000 plans from 2025 and used to score the due plans. All 1,200 predictions are in **`renewals_due_predictions.csv`** (column `p_terminated=Yes`).

## 3. The 10 plans most likely to leave

| # | Plan ID | Chance of leaving | Lives | Annual premium | Renewal increase | Account reviews (12m) | Service issues (12m) |
|---|---|---|---|---|---|---|---|
| 1 | GB-305069 | 74% | 14 | $61,387 | 16.8% | 0 | 0 |
| 2 | GB-729984 | 72% | 15 | $35,485 | 20.3% | 0 | 2 |
| 3 | GB-889741 | 69% | 13 | $37,383 | 16.7% | 0 | 0 |
| 4 | GB-343860 | 69% | 8 | $20,755 | 12.6% | 0 | 2 |
| 5 | GB-157168 | 65% | 9 | $24,346 | 10.5% | 0 | 2 |
| 6 | GB-899808 | 65% | 88 | $215,137 | 16.0% | 0 | 1 |
| 7 | GB-767778 | 65% | 27 | $66,809 | 15.4% | 0 | 0 |
| 8 | GB-876966 | 62% | 4 | $10,209 | 18.5% | 0 | 2 |
| 9 | GB-884169 | 62% | 78 | $180,826 | 10.5% | 0 | 0 |
| 10 | GB-176292 | 59% | 15 | $52,160 | 5.8% | not recorded | 1 |

Every one of them had no recorded account review last year, and nine of the ten face an increase above 10%. Plans 6 (GB-899808) and 9 (GB-884169) are the largest by premium.

## 4. What drives employers to leave

Ranked by how much the model relies on each factor. The leave rates are from last year's 6,000 plans (12.7% overall).

1. **Account reviews.** This is the strongest factor. Plans with no review last year left at **18.8%**; plans with one or more left at 7 to 9%.
2. **Plan size (premium and lives covered).** Small plans leave more often. The smallest quarter by premium left at 20%; the largest quarter at 8%.
3. **Renewal increase.** Increases up to 6% have little effect (10 to 11% left). Above 10%, the leave rate goes up to **27%**.
4. **Service issues.** 11% left with none, 16% with one, 23% with two or more.
5. **Smaller factors:** late payments (27% left with 2 or more), industry (Retail, Construction and Technology highest; Non-Profit lowest), newer clients (17% left in their first 2 years vs 10% after 10 years), and not using online admin.

Channel, billing method, funding type and the wellness program made almost no difference.

## 5. What-if tests on the three plans most likely to leave

For each plan, we changed one thing and asked the model again. "Account review" means one review held in the year before renewal.

| Plan | Today | + 1 account review | Increase cut to 8% | Increase cut to 5% | Service issues resolved | Review + 8% cap (+ issues resolved) |
|---|---|---|---|---|---|---|
| GB-305069 (16.8% increase, 2 late payments) | 74% | 48% | 55% | 49% | none open | **29%** |
| GB-729984 (20.3% increase, 2 service issues) | 72% | 39% | 54% | — | 54% | **14%** (all three) |
| GB-889741 (16.7% increase, new client) | 69% | 37% | 54% | 50% | none open | **26%** |

- For each plan, **an account review is the biggest single lever**. It lowers the chance of leaving by 26 to 33 points.
- Cutting the increase to 8% lowers it by 15 to 19 points.
- Resolving GB-729984's two service issues lowers it by 18 points.
- **Doing several together works best.** All three plans drop to between 14% and 29%.

**Important caveat.** These numbers show what the model learned about how plans that looked like this behaved last year. They are **not proven effects**. For example, employers who agree to an account review may already be more engaged and more likely to stay, and the review itself may not be what keeps them. A smaller increase may also hurt our margins on plans running loss ratios of 114 to 125%. Treat these numbers as a ranking of which actions are worth testing, not as a promise.

## 6. Recommended program: account reviews before renewal for high-risk plans with none

**What:** In the next few weeks, the account team holds one account review with each of the **200 highest-risk due plans that had no review last year**. Each of these plans has at least a 16% chance of leaving. Where the renewal increase is above 10% or a service issue is open, offer a reconsidered increase or a fix for the issue at the same meeting.

**Expected impact (model estimate):**

| Group | Expected departures now | With one review | Premium at risk now → with review |
|---|---|---|---|
| Top 100 highest-risk with no review | 39 | 18 | $3.76M → $1.71M |
| **Top 200 highest-risk with no review (recommended)** | **59** | **27** | **$5.87M → $2.53M** |
| All 533 due plans with no review (or none recorded) | 86 | 38 | — |

For the recommended 200 plans, that is about **32 fewer employers leaving and about $3.3M of annual premium kept**. Only reviews were changed in this estimate. Adding increase cuts or service fixes for the top plans should help more.

**How to prove it works:** because the effect is not proven, set aside a random 20 to 25% of the 200 plans and do not contact them early. Then compare renewal rates between the two groups in February. That will show what the reviews are really worth before we roll them out to the whole book.

## Files

- `clean.py`: the cleaning script (run with `python3 clean.py`)
- `renewals_2025.csv`, `renewals_due.csv`: the clean data
- `renewals_due_predictions.csv`: every due plan's predicted chance of leaving
- `scenario/`: the due plans rescored as if each had one account review (used only for section 6)
