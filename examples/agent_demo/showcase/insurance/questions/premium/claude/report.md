# Premium at risk at renewal: plans due Nov 2026 – Feb 2027

*Group Retention, prepared 22 Nov 2026*

## The short answer

- **1,200 plans** worth **$257.7M** in annual premium come up for renewal between 15 Nov 2026 and 15 Feb 2027.
- **Expected premium lost: about $21.5M (8.3% of the book due).** That's about 150 plans. The most likely outcome is $19M–$33M (90% range). The range is wide because a few very large plans could swing it either way, and because the model may understate how often large plans leave (see "How sure we are").
- **79 of these plans had renewal dates of 15–21 Nov**, so their dates have already passed. They hold $20.0M of premium and $1.4M of the expected loss. Their outcome may already be in the admin system, so look it up rather than relying on the model. **For renewals still ahead of us, the expected loss is $20.0M (90% range $16M–$24.5M before the calibration allowance).**
- **Where it concentrates:**
  - **By dollars:** Ontario, mid-to-large plans (50–249 lives) and Fully Insured business. Mostly this is because that's where the premium is.
  - **By rate** (more of each premium dollar at risk than average): small plans (under 25 lives, about 17%), Health & Dental-only plans (11.7%), Retail (11.4%) and Construction (10.3%).
  - **Accounts:** the top 100 accounts hold 36% of the expected loss. The top 240 (one in five plans) hold 57%. A retention effort aimed at a short list would cover most of the dollars.

## Where the $21.5M sits

"Expected at risk" adds up each plan's premium times its estimated chance of leaving. The 90% ranges show how much each group's actual loss could swing by chance alone, if the model's probabilities are right.

| Province | Plans | Premium due | Expected at risk | 90% range | % of premium |
|---|---|---|---|---|---|
| ON | 457 | $93.5M | **$8.0M** | $5.5–10.8M | 8.6% |
| QC | 257 | $53.5M | $4.3M | $2.6–6.3M | 8.0% |
| AB | 150 | $39.1M | $3.6M | $1.7–5.9M | 9.1% |
| BC | 185 | $38.9M | $2.9M | $1.7–4.7M | 7.5% |
| MB, SK, NS, NB, NL (combined) | 151 | $32.7M | $2.7M | — | 8.1% |

Province shares mostly mirror the premium split. The model makes almost no use of province (see below). The provinces' rates are statistically indistinguishable from each other.

| Industry | Plans | Premium due | Expected at risk | 90% range | % of premium |
|---|---|---|---|---|---|
| Professional Services | 276 | $54.1M | **$4.6M** | $2.9–6.5M | 8.4% |
| Retail | 169 | $34.9M | $4.0M | $2.3–6.0M | **11.4%** |
| Manufacturing | 202 | $41.8M | $3.1M | $1.8–4.8M | 7.5% |
| Construction | 133 | $28.7M | $3.0M | $1.5–4.8M | **10.3%** |
| Technology | 153 | $33.5M | $2.7M | $1.3–4.5M | 8.0% |
| Non-Profit / Public Sector | 148 | $37.1M | $2.2M | $1.0–4.0M | 5.9% |
| Health Care | 119 | $27.6M | $2.0M | $0.9–3.8M | 7.1% |

| Plan size (lives) | Plans | Premium due | Expected at risk | 90% range | % of premium |
|---|---|---|---|---|---|
| Under 25 | 374 | $17.1M | $2.9M | $2.3–3.4M | **16.8%** |
| 25–49 | 315 | $35.3M | $4.1M | $3.1–5.2M | 11.7% |
| 50–99 | 269 | $60.3M | $5.4M | $3.8–7.2M | 9.0% |
| 100–249 | 188 | $90.8M | **$6.3M** | $3.8–9.1M | 6.9% |
| 250+ | 36 | $50.5M | $2.4M | $0–5.7M | 4.8% |
| Headcount not recorded | 18 | $3.7M | $0.3M | — | 6.8% |

There are two ways to read the size table:

- **Small plans** leave most often.
- **Plans of 50–249 lives** carry 55% of the dollars at risk.
- **The 36 plans of 250+ lives** are the swing factor: one or two departures there would move the total by $1–3M.

Other cuts:

- **Funding:** Fully Insured $13.0M (9.6% of its premium), ASO $5.5M (5.8%), Pooled $2.9M (10.7%).
- **Plan type:** Health & Dental $9.3M (11.7%), Health, Dental & Life $6.6M (10.1%), Full Package $5.5M (4.9%).
- **Renewal month:** Dec $7.4M, Jan $7.1M, Feb (to the 15th) $3.7M, Nov $3.3M.

The 15 plans with the largest expected loss are below. Each plan's chance of leaving, and its expected loss, is in `renewals_due_predictions.csv` (column `p_terminated=Yes`).

| Plan | Prov | Industry | Lives | Premium | Chance of leaving | Renews |
|---|---|---|---|---|---|---|
| GB-104715 | AB | Retail | 491 | $1.53M | 24% | 16 Jan |
| GB-535804 | AB | Retail | 169 | $0.60M | 42% | 7 Dec |
| GB-883080 | ON | Construction | 319 | $0.91M | 25% | 27 Jan |
| GB-430907 | ON | Construction | 387 | $1.06M | 18% | 13 Feb |
| GB-383348 | QC | Prof. Services | 129 | $0.38M | 49% | 4 Feb |
| GB-502505 | ON | Health Care | 149 | $0.36M | 47% | 23 Dec |
| GB-274377 | SK | Manufacturing | 235 | $0.66M | 24% | 4 Dec |
| GB-361777 | AB | Technology | 405 | $1.15M | 12% | 27 Dec |
| GB-853268 | ON | Construction | 74 | $0.26M | 50% | 14 Dec |
| GB-829546 | BC | Manufacturing | 206 | $0.46M | 28% | 27 Nov |
| GB-420366 | QC | Prof. Services | 127 | $0.44M | 28% | 15 Dec |
| GB-394560 | ON | Prof. Services | 80 | $0.38M | 32% | 2 Feb |
| GB-899808 | BC | Retail | 88 | $0.22M | 55% | 6 Dec |
| GB-673163 | ON | Technology | 339 | $0.89M | 13% | 28 Nov |
| GB-537512 | QC | Construction | 118 | $0.44M | 27% | 20 Jan |

## How sure we are

**What the model learned from.** It's a tabular foundation model (TabPFN) trained on last year's 6,000 renewals, of which 12.7% left. Province, industry, size, plan type, funding, channel, billing, premium, renewal increase, loss ratio, payment and service history, account reviews, tenure, online admin, wellness program and app use were all available to it.

**How well it ranks plans.** On renewals it didn't see, it ranks leavers above stayers 74% of the time (ROC AUC 0.74; scikit-learn gradient boosting scored 0.71). That's useful for ordering a call list but far from certain. Only 16 of the 1,200 due plans have a better-than-even chance of leaving. A typical leaver looks like a plan with a 20–40% chance, not a sure thing. **Treat these as probabilities, not a list of who will leave.**

**Whether its probabilities add up.** I held out 1,800 of last year's plans and compared:

- **Leaver count:** the model expected 232 and 228 actually left. Within each fifth of the risk ranking, predicted and actual rates agree to within 2 points.
- **Premium lost:** actual was **18% higher** than the model expected ($37.9M vs $32.2M). Most of the gap came from plans of 100+ lives, where 35 left against 25 expected. The 90% interval on that ratio is 0.97–1.40, so it could be chance. The full-history leaving rates by size (100–249 lives: 8.3%; 250+: 5.7%) match what the model assigns the due plans (7.6% and 5.8%).
- **How I handled it:** I kept $21.5M as the central figure and widened the range to $19M–$33M to allow for it. **If you need a cautious planning number, use about $25M.**

**Limits.**

- Each plan's chance is estimated separately. A market-wide event, such as a competitor's pricing push or a broker moving a book of business, would make departures cluster, and none of these ranges allow for it.
- Last year's patterns are assumed to hold this year.

## Which plan traits go with leaving

All of these are **patterns in last year's data and in what the model relies on, not proven effects**. None of it shows that changing a trait would change whether the employer stays. Rates below are last year's actual leaving rates, with 95% intervals.

1. **No account review in the past 12 months** is the model's strongest signal. Plans with no review left at 18.8% (17.3–20.4%), against 8.0% (6.9–9.3%) for plans with one review. More than one review shows no further benefit: 8.1% with two or more vs 8.0% with exactly one. That comparison could have detected a difference of about 2.7 points (80% power). Reviews may simply go to healthier, better-managed accounts, so this doesn't prove reviews retain clients. It is the most obvious thing to test, for example by offering reviews to a random half of unreviewed at-risk plans.
2. **Late payments and service issues.** Leaving rises with each late payment: 10.6% with none, 16.7% with one, 24.5% with two, 37% with three (n=68, interval 26–49%). It also rises with service issues: 10.7% with none, 16.4% with one, 23.9% with two.
3. **Size of the renewal increase.** Up to about 8%, leaving creeps up slowly, from 9.3% (0–2% increase) to 12.0% (6–8%). Above that it climbs: 17% at 8–10% (interval 12–24%), 22% at 10–12%, and 27–28% above 12%. The data put the step somewhere between 8% and 12%. With only about 150 plans in each band from 8% to 12%, they can't pin it down more precisely. This is also only a pattern: large increases go to plans with bad claims experience, and those plans may be shopping around anyway.
4. **Loss ratio at or above 100%:** 25.3% left (22.5–28.4%), against about 10–13% for everything below 100%.
5. **Cheque/invoice billing** (21.1%), against 11–12% for pre-authorized debit or EFT.
6. **Smaller and newer plans.**
   - **Size:** 10–24 lives 18.5%, 1–9 lives 15.4%, falling to 5.7% for 250+.
   - **Tenure:** under 2 years about 17%, 10+ years 10%.
   - **Plan type and funding:** Health & Dental-only plans (16.0%) and Pooled or Fully Insured funding (14–16%) leave more than Full Package (8.9%) or ASO (7.3%).
7. **Traits that showed no detectable difference:**
   - **Sales channel:** Broker 13.1%, Consultant 12.2%, Direct 12.5%. Differences of about 2.5–3 points could have been detected.
   - **Province:** in Ontario, Quebec, BC and Alberta, a difference of about 2.5–3.7 points could have been detected and none was found. The smaller provinces have too few plans: only differences of 6–13 points could have been detected there, so moderate differences can't be ruled out. Treat the high Saskatchewan (17.8%) and low New Brunswick (6.1%) rates as unconfirmed.
   - **Online admin and wellness program:** plans without them leave about 3 points more often (significant on its own). The model gets almost nothing extra from these once it knows the other traits, so they probably go along with other traits rather than mattering in themselves.

## Changes to the plan, and data decisions you should know about

- **Step 1 (cleaning) found more than format problems.** `clean.py` handles all of these:
  - **Rows removed:** 120 exact duplicate rows in the history and 12 in the due file, 6 `TEST-` plans, and a `TOTAL` row ($1.27B) at the bottom of the history.
  - **Headcounts blanked:** placeholder lives counts (−1, 0, 99999, 999999). Plans with blank headcounts were kept.
  - **Units fixed:** loss ratios came both as fractions (0.856) and as percentages (85.6%). Renewal increases came with and without a "%" sign but were on the same scale.
- **"In force" status (896 plans, 15% of the history) was counted as renewed.** The plan was still with us after its 2025 renewal date. "Terminated", "TERM" and "moved carrier" were counted as left. **Please confirm with plan administration.** If "In force" meant something else, the leaving rate and every figure here would change.
- **Renewal window.** The due file runs from 15 Nov 2026 to 15 Feb 2027, so it was pulled about a week ago and includes 79 renewal dates that have already passed (reported separately above).
- **Step 2:** besides the standard accuracy check, I added the hold-out check above on whether the probabilities add up to the actual premium lost, because the headline is a sum of probabilities.
- **The model didn't use calendar dates directly,** only tenure, the years from plan start to renewal.

## Files

- `clean.py`: cleaning script. Run `python clean.py` to rebuild `renewals_2025.csv` and `renewals_due.csv`.
- `renewals_due_predictions.csv`: one row per due plan, with `p_terminated=Yes`, its estimated chance of leaving.
- `work/`: the hold-out split and its predictions, used for the calibration check.
- Fitted models: `renewal_churn` (all 2025 data, used for the due plans) and `holdout_check` (calibration only).
