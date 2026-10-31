You are a retention analyst at a telecom company. In this directory:

- `customers_history.csv`: past customers, with `Churn` (Yes/No): did they leave?
- `customers_current.csv`: today's customers. Nobody knows yet who will leave.

You have tabular foundation models (the tabctx tools) for predictions
from tables. Use them, not a model you write yourself.

1. Find out how well churn can be predicted from this data, compared with
   a standard gradient-boosting model, and pick the better tabctx model.
2. Predict the churn risk of every current customer.
3. Find which factors drive churn.
4. Take the three highest-risk current customers on a month-to-month
   contract. For each, test whether moving them to a one-year contract
   would lower their predicted risk, and by how much.
5. Write `report.md`: how accurate the model is, the 10 highest-risk
   customers (id and probability), the main drivers, the contract what-if
   results, and one recommended retention action with expected impact.

Keep the predictions CSV the tools write: it will be scored against what
the customers actually did.
