# Account review meetings for plans renewing soon: are they worth it, and for whom?

## Short answer

**Yes, probably, but not for "every at-risk plan".** Hold the reviews for plans that **have had no account review in the last 12 months** (or where we don't know). Among those, start with the plans whose chance of leaving is highest.

- In last year's data, plans with no review left at **18.8%**. Plans with at least one review left at **about 8%**. After comparing like with like (plan size, renewal increase, funding type and tenure), the gap is still **9.3 points** (95% range 7.3 to 10.8).
- **A second or third review shows no extra benefit.** Plans with one review left at 8.0%; plans with two or more left at 8.1%. More on this below.
- So **62 of the plans the model rates high-risk (20%+ chance of leaving) have already had a review**. By this data, another meeting is unlikely to change their outcome. Their risk comes mostly from steep renewal increases, late payments and service issues. Those need a pricing or service fix, not another meeting.
- **The list I recommend:** `review_targets.csv`. It holds 353 plans with zero (or unrecorded) reviews where the model expects a review to cut the chance of leaving by at least 5 points. **25 of them have a renewal date that has already passed** (Nov 15–21), so **328 can actually be acted on**. If the model's effect is real, reviewing them would keep about **39 more plans** and about **$4.2M of annual premium**. That works out to roughly $12k of premium per meeting.

**Important:** the effect of a review here is a **pattern the model learned from past data, not a proven effect**. See "How much to trust this" below.

## What I did, step by step

### 1. Standardizing the exports (`clean.py` → `renewals_2025.csv`, `renewals_due.csv`)
- Removed a `TOTAL` footer row and 120 exact duplicate rows (12 in the due file). That leaves **6,000 past plans and 1,200 due plans**, one row per plan. No plan appears in both files.
- Unified provinces (e.g. "Ont.", "ontario" → ON; "PQ", "Québec" → QC), industries, plan types, funding, channel, billing, and yes/no fields (Y/1/TRUE/yes).
- Parsed four date formats and five currency/percent formats. Loss ratio came both as "85.6%" and as "0.856"; both are now percent.
- Set placeholder headcounts (-1, 0, 99999, 999999) to blank. Set "n/a", "unknown" and "-" review counts to blank (242 past plans, 57 due plans).
- Added `tenure_years` (time from plan start to renewal) and `premium_per_life`.
- **`terminated`:** Renewed / RENEWED / **In force** → No. Terminated / TERM / moved carrier → Yes. I read "In force" (896 plans) as the plan carrying on, i.e. renewed. **Please confirm that with the admin team.** Result: 764 of 6,000 plans left (12.7%).

### 2. How well past data predicts who leaves
On 1,500 past plans held back for testing, TabPFN ranks leavers from stayers with **ROC AUC 0.76**. TabICL scored about the same (0.76), and a standard gradient-boosting model scored 0.74. That is useful for ranking, but far from certain. Few plans have a predicted chance above 50%, so treat the scores as a priority order, not as yes/no calls. (Accuracy was 87%, but that is no better than guessing "everyone renews".) The average predicted risk for the due plans is 12.5%, close to last year's actual 12.7%.

### 3. Scores for the plans renewing soon
All 1,200 plans are scored in **`renewals_due_predictions.csv`** (column `p_terminated=Yes`).

### 4. Traits most linked to leaving (patterns, not causes)
Ranked by how much the model's accuracy drops when the trait is scrambled:
1. **Account reviews in the last 12 months.** None = much higher risk.
2. **Renewal increase.** Above 10%, leaving rates jump: 41% for plans with no review, 15% for plans with one.
3. **No online admin**, **service issues**, **late payments.** Each additional issue or late payment goes with higher risk.
4. Industry, tenure and plan size. Newer and smaller plans leave more.

Province, channel and funding type add little once the other traits are known. These are associations. For example, late payments may signal an employer in trouble rather than cause anything.

### 5. How much a review might lower each plan's risk (the step I changed)
I re-scored every due plan as if it had one more review and compared that with its current score:

| Reviews so far | Plans | Risk now | Risk with one more review | Drop |
|---|---|---|---|---|
| 0 | 476 | 17.9% | 9.2% | **8.7 pts** |
| unknown | 57 | 14.8% | 9.6% | 5.3 pts |
| 1 | 382 | 9.2% | 9.0% | 0.3 pts |
| 2 or more | 285 | 7.5% | 7.1% | 0.4 pts |

For plans with no review, the drop grows with risk: about 3 points for low-risk plans and about 19 points for plans above 30% risk. That is why I changed the plan. Ranking by risk alone would give a quarter of the 200 riskiest plans a meeting the data says won't help. So the list is ranked by **expected drop in risk**, not by risk.

**"A second review makes no difference": how sure is that?** Plans with one review left at 8.0% (1,973 plans); plans with two or more left at 8.1% (1,307 plans). After comparing like with like, the 95% range runs from a second review *lowering* the leaving rate by 1.4 points to *raising* it by 3.0 points. With this many plans, we could reliably have detected a true effect of about **2.7 points**. A small benefit under that size can't be ruled out.

### 6. Recommendation
1. **Review the 328 actionable plans in `review_targets.csv`**, in the order given. The first 100 have the biggest expected drops (13 to 30 points) and account for about 18 of the 39 expected saves. If meeting slots are scarce, the first 150–200 capture most of the value.
2. **Book the December renewals first.** 117 targets renew in December and 63 in late November. A review held in the new year comes too late for them. If "next quarter" means January–March, only the ~173 January–February renewals still qualify.
3. **For the 57 plans with no recorded review count, check the records first.** If they really had no review, they belong on the list.
4. **For high-risk plans that have already had a review**, look at the price increase and the open service problems instead of booking another meeting.
5. **Is it worth it?** About $12k of annual premium kept per meeting, *if* the effect is real. Premium is not profit, so compare that against what a meeting costs (staff time plus any concessions offered in it). Even if the true effect were only a quarter of what the model suggests, it would still be about $3k of premium per meeting.

## How much to trust this
- **The review effect is a pattern, not a proven cause.** Reviews weren't assigned at random. The gap held up after comparing like with like on size, renewal increase, funding and tenure. But it could still be partly something else. An employer that has already decided to leave may turn down meetings, and well-run relationships get both more meetings and more loyalty. If so, the real benefit is smaller than 9 points.
- **To find out for sure:** randomly hold back about 1 in 3 of the lower-priority targets and compare their leaving rate with the plans that got a review. One quarter of this will only show a large effect (roughly 13–15 points), so keep the holdout running for several quarters to get a firm answer.
- The model ranks reasonably well (AUC 0.76) but misses many leavers. Expect some plans not on the list to leave.

## Files
- `clean.py` → `renewals_2025.csv`, `renewals_due.csv`: cleaned tables.
- `renewals_due_predictions.csv`: risk score for all 1,200 due plans.
- `review_targets.csv`: recommended review list with risk now, risk with a review, expected drop, premium, and whether the renewal date has already passed.
- `whatif/`: the re-scoring scenarios behind step 5.
