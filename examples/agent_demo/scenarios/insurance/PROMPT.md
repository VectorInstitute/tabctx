You are an analyst on the customer retention team at Maple Mutual, a
Canadian insurer. In this directory are two raw exports from the policy
system:

- `renewals_2025_export.csv`: last year's renewals, with
  `Status at Renewal`: did the policyholder renew or cancel?
- `renewals_due_export.csv`: policies whose renewal is due in the next
  three months. Nobody knows yet who will cancel.

You have tabular foundation models (the tabctx tools) for predictions
from tables. Use them, not a model you write yourself.

1. Clean both exports with one Python script, `clean.py` (pandas is
   installed), and run it. Write `renewals_2025.csv` and
   `renewals_due.csv` with snake_case column names (`Policy ID` becomes
   `policy_id`), one row per policy, consistent categories, numbers as
   numbers, missing values left empty, and `years_as_customer` at
   renewal in place of the two dates. In the history, `cancelled` is
   Yes or No. Keep every policy in `renewals_due.csv`. List each problem
   you found and what you did about it.
2. Find out how well cancellation can be predicted from the clean data,
   compared with a standard gradient-boosting model, and pick the better
   tabctx model.
3. Predict every due policyholder's chance of cancelling.
4. Find which factors drive cancellation.
5. Take the three policyholders most likely to cancel. For each, test
   two or three things the retention team could do (for example, an
   advisor check-in call, which counts as one advisor contact, moving
   them to pre-authorized debit, or a smaller premium increase) and how
   much each moves their chance of cancelling.
6. Write `report.md` for the head of retention, in plain language: what
   was wrong with the data and how it was fixed, how accurate the model
   is, the 10 policyholders most likely to cancel (id and probability),
   the main drivers, the what-if results, and one recommended retention
   program with its expected impact. Say plainly where the what-if
   results are patterns the model learned, not proven effects.

Keep the predictions CSV the tools write: it will be scored against who
actually cancelled.
