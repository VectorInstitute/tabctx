# Which employers are likely to leave at renewal, why, and what might keep them

*For: Head of Group Retention. Prepared 22 November 2026 from the plan-administration exports.*

## The short version

- **About 130–150 of the 1,200 plans renewing between mid-November and mid-February are likely to leave.** The two models we used give those two figures. The higher one is probably closer, because the main model ran about 3 points low when we tested it on last year's data. That is roughly $18M of annual premium at risk, out of $258M due.
- **The risk is concentrated.** 85 plans have a 30% or higher chance of leaving. On last year's data, calling the top 20% of the list would have reached about **46% of the employers who left**, and about one in three of the plans called would actually have left.
- **The strongest warning signs**, in order of how much the model relies on them:
  1. no account review in the past year
  2. small plans, measured by premium or lives covered
  3. a large renewal increase, which goes with a high loss ratio
  4. service issues
  5. late payments
  6. paying by cheque or invoice
- **What looks most promising to act on: make sure every plan gets at least one account review before renewal.** In last year's data, plans with no review left at 19%, against 8% for plans with one. The gap holds for small and large plans and for good and bad loss ratios. **A second or third review showed no further benefit.** 71 of the 85 highest-risk plans due now have had no review.
- **Capping renewal increases is not a good deal on these numbers.** The plans with big increases mostly have loss ratios over 100%. A 10% cap would give up about $1.6M of increase revenue to keep, on average, about $1.5M of premium on plans that are already losing money.
- **These are patterns the model learned from last year, not proven effects.** The data cannot tell us whether reviews keep employers, or whether account managers simply review the accounts that are going to stay. The way to find out is a simple randomized test (see "What I recommend").

---

## 1. Cleaning the exports (`clean.py`)

`python clean.py` reads the two raw exports and writes `renewals_2025.csv` (6,000 plans) and `renewals_due.csv` (1,200 plans).

| Problem in the export | What I did |
|---|---|
| 120 exact duplicate rows in the history and 12 in the due file | Removed. No plan had two *different* rows. |
| 6 `TEST-000x` accounts and a `TOTAL` footer row in the history | Removed. They aren't real plans. |
| Dates in four formats (`2025-03-03`, `2025/03/03`, `Mar 03, 2025`, `03-Mar-2025`) | Converted to `YYYY-MM-DD`. Added `tenure_years`, the years from the plan's start to its renewal. |
| Loss ratio sometimes given as a percent (`85.0%`) and sometimes as a fraction (`0.85`) | Converted all to percent. |
| Premiums written as `$74,848.88`, `CAD 9,676` or `62770.46` | Converted all to plain numbers. |
| 34 spellings of the province, 21 of industry, 11 of plan type, 6 of funding, 9 of channel, 10 of billing, 10 of yes/no | Mapped to one label each. If a new spelling appears, the script stops with an error instead of guessing. |
| Lives covered of `-1` or `999999`, plus `n/a`, `unknown` and `-` in several columns | Treated as missing: 113 values of lives covered, 242 of account reviews and 279 of app use in the history. |
| Outcome labels `Renewed`, `RENEWED`, `In force`, `Terminated`, `TERM`, `moved carrier` | `terminated` = **Yes** for Terminated, TERM and moved carrier (764 plans, 12.7%). **No** for the rest. |

**A judgment call you should know about.** I counted the 896 "In force" plans as renewed. They are spread evenly across every renewal month of 2025, so they are not a backlog of undecided renewals. They look like plans that are simply still active. If "In force" means something else in the admin system, the leave rate and the model would change.

## 2. How well last year predicts who leaves

I held out a quarter of last year's plans and tested the models on them:

| Model | ROC AUC (0.5 = coin flip, 1 = perfect) |
|---|---|
| TabICL (the model used for the scores) | 0.77 |
| TabPFN (second opinion) | 0.76 |
| Standard gradient boosting, for comparison | 0.74 |

The scores are **good for ranking, not for certainty.** Even among the plans the model is most worried about, about half renew anyway. The model also misses some leavers entirely: about 40% of leavers fall outside the top 30% of the list.

Results on the held-out plans:

| If you contact the top… | …you reach this share of leavers | Share of contacted plans that actually left |
|---|---|---|
| 5% of plans | 18% | 51% |
| 10% | 30% | 42% |
| 20% | 46% | 32% |
| 30% | 58% | 27% |

The two models rank the due plans almost identically: rank correlation 0.985, and 89 of the same plans in each model's top 100.

## 3. Plans due for renewal: who is at risk

Every plan's score is in **`renewals_due_predictions.csv`**. The column `p_terminated=Yes` is the chance of leaving.

- Expected leavers: **131** (TabICL) to **151** (TabPFN). On held-out data TabICL predicted 10.7% would leave when 13.8% did, so treat 150 as the more realistic figure.
- 186 plans have a 20% or higher chance of leaving, and 85 plans have 30% or higher.
- Risk is spread evenly across the four months: 13, 29, 27 and 16 high-risk plans in Nov, Dec, Jan and Feb.
- **Check this:** the "due" export includes 79 plans with renewal dates of 15–21 November, which have already passed, and 6 more renewing today. Six of those 79 are high-risk, including GB-389851 (Manufacturing, $162K premium, 47%). Their outcome may already be known.

**Highest chance of leaving:**

| Plan | Renews | Industry | Lives | Premium | Increase | Loss ratio | Reviews (12m) | Service issues | Late pmts | Chance of leaving |
|---|---|---|---|---|---|---|---|---|---|---|
| GB-305069 | 2027-01-03 | Construction | 14 | $61K | 16.8% | 114% | 0 | 0 | 2 | 74% |
| GB-729984 | 2027-02-07 | Construction | 15 | $35K | 20.3% | 125% | 0 | 2 | 0 | 72% |
| GB-343860 | 2026-11-29 | Technology | 8 | $21K | 12.6% | 110% | 0 | 2 | 0 | 69% |
| GB-889741 | **2026-11-22** | Technology | 13 | $37K | 16.7% | 120% | 0 | 0 | 1 | 69% |
| GB-157168 | 2026-12-08 | Manufacturing | 9 | $24K | 10.5% | 102% | 0 | 2 | 0 | 65% |
| GB-899808 | 2026-12-06 | Retail | 88 | $215K | 16.0% | 112% | 0 | 1 | 0 | 65% |
| GB-767778 | 2027-01-19 | Manufacturing | 27 | $67K | 15.4% | 115% | 0 | 0 | 1 | 65% |
| GB-876966 | 2027-01-31 | Manufacturing | 4 | $10K | 18.5% | 120% | 0 | 2 | 1 | 62% |
| GB-884169 | 2027-02-07 | Technology | 78 | $181K | 10.5% | 105% | 0 | 0 | 1 | 62% |
| GB-176292 | 2027-01-03 | Retail | 15 | $52K | 5.8% | 72% | missing | 1 | 2 | 59% |
| GB-853268 | 2026-12-14 | Construction | 74 | $260K | 16.2% | 114% | 0 | 1 | 2 | 54% |
| GB-346023 | **2026-11-22** | Technology | 7 | $22K | 14.1% | 116% | 0 | 0 | 0 | 53% |
| GB-510578 | 2026-12-31 | Health Care | 7 | $24K | 15.2% | 116% | 0 | 1 | 0 | 51% |
| GB-585362 | 2027-02-08 | Technology | 34 | $81K | 10.0% | 105% | 1 | 1 | 1 | 51% |
| GB-769302 | 2026-12-23 | Construction | 35 | $104K | 5.2% | 73% | 0 | 1 | 2 | 50% |

**Most premium at risk** (chance of leaving × premium). These are the large accounts worth a senior call even though their chance of leaving is moderate:

| Plan | Renews | Industry | Lives | Premium | Increase | Loss ratio | Reviews | Chance | Premium at risk |
|---|---|---|---|---|---|---|---|---|---|
| GB-104715 | 2027-01-16 | Retail | 491 | $1.53M | 9.8% | 106% | 1 | 19% | $296K |
| GB-535804 | 2026-12-07 | Retail | 169 | $598K | 13.3% | 111% | 0 | 42% | $252K |
| GB-883080 | 2027-01-27 | Construction | 319 | $906K | 15.6% | 116% | 0 | 21% | $194K |
| GB-383348 | 2027-02-04 | Prof. Services | 129 | $380K | 13.6% | 106% | 0 | 49% | $186K |
| GB-502505 | 2026-12-23 | Health Care | 149 | $361K | 14.6% | 110% | 0 | 49% | $177K |
| GB-274377 | 2026-12-04 | Manufacturing | 235 | $660K | 13.4% | 107% | 2 | 26% | $169K |

What the 85 plans at 30% or higher have in common:

| Feature | Plans out of 85 |
|---|---|
| No account review (or none recorded) | 71 |
| At least one service issue | 49 |
| Loss ratio above 100% | 50 |
| 25 lives or fewer | 49 |
| Renewal increase above 10% | 46 |
| At least one late payment | 42 |
| Pay by cheque or invoice | 37 |

## 4. Why: what the model leans on

The table below shows the leave rates in last year's data, with 95% ranges. **These are associations.** They describe who left; they do not prove what made them leave.

| Feature | Leave rate last year |
|---|---|
| **Account reviews in past 12m** | 0 reviews: **18.8%** (17.2–20.3)<br>1 review: 8.0%<br>2 reviews: 8.9%<br>3+ reviews: 6.6% |
| **Renewal increase** | 0–2%: 9.3%<br>4–6%: 11.7%<br>8–10%: 17%<br>10–12%: 22%<br>12–15%: 27%<br>over 15%: **28%** |
| **Loss ratio** | under 70%: 9.5%<br>90–100%: 13%<br>100–110%: 21%<br>over 110%: **28%** |
| **Plan size (lives)** | 10–25 lives: **18%**<br>50–100 lives: 9%<br>over 250 lives: 6% |
| **Service issues** | 0: 10.7%<br>1: 16.4%<br>2 or more: **23%** |
| **Late payments** | 0: 10.6%<br>1: 16.7%<br>2 or more: **27%** |
| **Billing** | Cheque/invoice: **21%**<br>Pre-authorized debit: 11%<br>EFT: 12% |
| **Funding** | ASO: 7%<br>Fully insured: 14%<br>Pooled: 16% |
| **Tenure** | Under 2 years: about 17%<br>Over 10 years: 10% |

When each column was shuffled to measure how much the model relies on it, the order was:

1. account reviews
2. premium
3. renewal increase
4. service issues
5. lives covered

After those, industry, late payments, tenure and online admin had smaller effects.

**Two cautions on reading these:**

1. **The renewal increase and the loss ratio largely move together** (correlation 0.79). Our pricing sets the increase from the claims experience. Last year, *no* plan with a loss ratio over 110% got an increase of 10% or less. So the data mostly cannot separate "the increase drove them away" from "a bad claims year drove them away." There is one band where the two do overlap: plans with loss ratios of 100–112%. In that band, plans given an increase over 10% left at 25.7% (n=311), against 10.8% (n=83) for those given 10% or less. The difference is **+15 points, with a 95% range of +7 to +23 points**. The data could only have reliably detected a gap of 15 points or more in this group, so this is suggestive, not firm.
2. **The review pattern could run in either direction.** Account managers may skip reviews for employers who are already disengaged or have given notice. The pattern does hold inside every size band and every loss-ratio band, which rules out size and claims as the explanation. It cannot rule out reverse causation.

**Where the data could detect a difference and did not find one.** The "smallest detectable" column is the gap the data had an 80% chance of finding if it were real:

| Comparison | Difference in leave rate (95% range) | Smallest detectable |
|---|---|---|
| 2 or more reviews vs 1 review | −0.1 pts (−2.0 to +1.8) | 3.3 pts |
| Increase of 3% or less vs 3–6% | −1.3 pts (−3.1 to +0.5) | 2.8 pts |
| Broker vs consultant or direct | +0.7 pts (−1.0 to +2.4) | 2.4 pts |
| EFT vs pre-authorized debit | +1.0 pts (−0.9 to +2.9) | 2.9 pts |

So a second review, the sales channel, the EFT-versus-PAD choice and small changes among modest increases make **little or no difference**. A real effect of 2–3 points could still be hiding in any of these.

**The threshold on increases.** Leave rates are fairly flat up to about 8% (9–12%), then rise to 17% at 8–10% and 22% or more above 10%. The size of the 8–10% group (n=153) means the jump could lie anywhere between about 8% and 12%. And as caution 1 explains, it coincides with loss ratios reaching about 100%.

## 5. What could keep them: what-if tests

For each idea, I changed the relevant column for every due plan it would apply to and re-scored the plans with both models. **This tells us what the model expects given last year's patterns, not what the action will actually do.**

| Scenario | Plans changed | Avg change in chance of leaving (TabICL / TabPFN) | Expected leavers kept (TabICL / TabPFN) |
|---|---|---|---|
| Every plan with no review gets **one** account review | 533 | −9.0 / −8.1 pts | **48 / 43** |
| Every plan gets at least **two** reviews | 915 | −4.5 / −5.3 pts | 41 / 48 |
| …of which, plans going from 1 review to 2 | 382 | +0.8 / −0.6 pts | about 0 |
| Fix all service issues (no issues at all) | 360 | −6.6 / −6.3 pts | 24 / 23 |
| Move cheque/invoice payers to pre-authorized debit | 196 | −6.3 / −5.9 pts | 12 / 12 |
| Turn on online admin | 457 | −3.0 / −3.5 pts | 14 / 16 |
| Add a wellness program | 665 | −1.9 / −2.5 pts | 12 / 16 |
| Cap the renewal increase at 10% ⚠ | 150 | −6.0 / −5.3 pts | 9 / 8 |
| Cap the renewal increase at 6% ⚠ | 283 | −5.4 / −5.8 pts | 15 / 17 |
| Take 3 points off every increase over 6% ⚠ | 283 | −3.6 / −3.1 pts | 10 / 9 |

How to read the table:

- **The two models agree closely on every scenario.** That shows the result is not a quirk of one model. It does *not* show that the effects are real: both models learned from the same data, so both carry the same reverse-causation risk.
- **The review result has a clear threshold: getting to one review is what matters.** For plans that already have one review, adding a second changes the model's estimate by less than a point, in either direction. That matches the raw data (1 vs 2 or more reviews: −0.1 ± 1.9 pts).
- **⚠ The price-cap scenarios go beyond what the data has seen.** Two-thirds of the plans affected by the 10% cap have loss ratios over 110%, and last year no such plan had an increase that low. The model is extrapolating here.
- **Price caps also cost money.**

  | Cap | Increase revenue given up | Premium the model expects to keep | Average loss ratio of affected plans |
  |---|---|---|---|
  | 10% | about $1.6M | about $1.5M | 113% |
  | 6% | about $3.3M | about $2.6M | 100% |

  Under both caps, keeping these plans at a lower rate is likely to lose money on claims. Price concessions only make sense one plan at a time, for large accounts where we have reason to think the claims year was a one-off.
- **Fixing service issues and moving payers to PAD are probably partly symptoms**, not levers. An employer that is already unhappy or struggling financially may raise issues and pay by cheque. These are still sensible things to do. Just don't count on the modelled savings.

## 6. What I recommend

1. **Do now (December–February renewals): call the high-risk list.** Start with the 85 plans at 30% or higher, in renewal-date order, plus the six large accounts in the "most premium at risk" table. Make sure each gets an account review before its renewal date. Give priority to the 71 high-risk plans with no review this year, and to those with open service issues.
2. **Check the 79 plans whose renewal date has already passed (15–21 Nov)** against the admin system, starting with the six high-risk ones.
3. **Test whether reviews actually work.** The model credits reviews with keeping about 45 employers this cycle, but that may be reverse causation. Split the plans with no review that are not on the urgent list at random: half get a review, half get the usual treatment.
   - With about 230 plans in each half, the test would detect a drop of roughly 10 points, from about 19% leaving to about 9%. That is about the size of gap last year's data suggests.
   - A smaller real effect would need more plans, or pooling over two renewal cycles.
   - The result tells us whether to fund reviews for every account.
4. **Don't cap increases broadly.** For high-loss-ratio plans, a conversation about plan design (deductibles, drug formulary, moving to ASO) is likely better than a straight discount. That suggestion is a judgment, not something this data tested.
5. **Low-cost hygiene:** move cheque payers to PAD and switch on online admin. These cost little and point the right way, even if the effect is uncertain.

## Changes to the approved plan

- **Step 1:** I counted "In force" as renewed (see section 1). I removed the TEST accounts and the TOTAL row, which the plan did not mention. I replaced the two date columns with tenure for modelling, because the due plans' 2026–27 dates are outside last year's range.
- **Step 2:** Besides the accuracy check, I measured how many leavers the top of the list captures, which is more useful for planning calls.
- **Step 5:** Because increase and loss ratio move together, the increase what-ifs extrapolate. I added a check within the one loss-ratio band where the two overlap, and set the cost of a cap against the premium it keeps. I also ran every scenario on a second model (TabPFN) to check that the answers don't depend on one model.

## Files

- `clean.py`: rebuilds `renewals_2025.csv` and `renewals_due.csv` from the raw exports.
- `renewals_due_predictions.csv`: every due plan with `p_terminated=Yes`, the TabICL model trained on all 6,000 plans from 2025.
- `scenarios/`: the what-if input files and their scores from both models, the TabPFN baseline scores (`pfn_baseline.csv`), and the held-out test files.
- Saved models: `renewal_churn` (TabICL, the main one), `renewal_churn_pfn` (TabPFN) and `holdout_check` (the held-out test). The models leave out plan ID and the two raw date columns.
