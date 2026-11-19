You are an analyst on the group retention team at Maple Mutual Group
Benefits, a Canadian insurer that sells group health, dental, life and
disability plans to employers. In this directory are two raw exports
from the plan administration system:

- `renewals_2025_export.csv`: last year's plan renewals, with
  `Renewal Status`: did the employer renew its plan with us or leave?
- `renewals_due_export.csv`: plans whose renewal is due in the next
  three months. Nobody knows yet who will leave.

You have tabular foundation models (the tabctx tools) for predictions
from tables. Use them, not a model you write yourself.

1. Clean both exports with one Python script, `clean.py` (pandas is
   installed), and run it. Write `renewals_2025.csv` and
   `renewals_due.csv` with snake_case column names (`Plan ID` becomes
   `plan_id`), one row per plan, consistent categories, numbers as
   numbers, missing values left empty, and `years_with_us` at renewal in
   place of the two dates. In the history, `terminated` is Yes (the
   employer left) or No. Keep every plan in `renewals_due.csv`. List each
   problem you found and what you did about it.
2. Find out how well leaving can be predicted from the clean data,
   compared with a standard gradient-boosting model, and pick the better
   tabctx model.
3. Predict every due plan's chance of leaving.
4. Find which factors drive employers to leave.
5. Take the three plans most likely to leave. For each, test two or
   three things the retention team could do (for example, an account
   review meeting, which counts as one account review, a smaller renewal
   increase, or resolving their open service issues) and how much each
   moves their chance of leaving.
6. Write `report.md` for the head of group retention, in plain language:
   what was wrong with the data and how it was fixed, how accurate the
   model is, the 10 plans most likely to leave (id and probability), the
   main drivers, the what-if results, and one recommended retention
   program with its expected impact. Say plainly where the what-if
   results are patterns the model learned, not proven effects.

Keep the predictions CSV the tools write: it will be scored against who
actually left.
