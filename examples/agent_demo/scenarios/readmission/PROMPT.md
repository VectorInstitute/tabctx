You are an analyst on a hospital's care-transitions team. In this directory:

- `patients_history.csv`: past hospital stays of diabetic patients, with
  `readmitted_30d` (Yes/No): were they readmitted within 30 days?
- `patients_current.csv`: patients being discharged now. Nobody knows yet
  who will be readmitted.

You have tabular foundation models (the tabctx tools) for predictions
from tables. Use them, not a model you write yourself.

1. Find out how well readmission can be predicted from this data,
   compared with a standard gradient-boosting model, and pick the better
   tabctx model.
2. Predict the readmission risk of every current patient.
3. Find which factors drive readmission.
4. Take the three highest-risk current patients who are being discharged
   home (`discharge_disposition` = "Home"). For each, test two or three
   changes a care team could make (for example, discharge with home
   health service, `discharge_disposition` = "Home health service", or a
   change to their diabetes medication) and how much each moves their
   predicted risk.
5. Write `report.md`: how accurate the model is, the 10 highest-risk
   patients (id and probability), the main drivers, the what-if results,
   and one recommended follow-up program with its expected impact. Say
   plainly where the what-if results are associations the model learned,
   not proven effects.

Keep the predictions CSV the tools write: it will be scored against what
actually happened to these patients.
