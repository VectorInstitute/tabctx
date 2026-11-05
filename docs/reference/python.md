# Python API

The public API is importable from `tabctx`. Generated from the docstrings.

## Estimators

::: tabctx.estimators.TabctxClassifier
    options:
      members: [fit, predict, predict_proba]

::: tabctx.estimators.TabctxRegressor
    options:
      members: [fit, predict]

## Client

::: tabctx.client.TabctxClient
    options:
      members:
        - fit
        - fit_file
        - predict
        - predict_file
        - fit_predict
        - upload_csv
        - fit_uploaded
        - datasets
        - delete
        - models
        - tabular_models
        - default_model
        - limits
        - ready

::: tabctx.client.PredictResult

## Engine

::: tabctx.engine.TabctxEngine
    options:
      members: [fit, predict, fit_predict, evict, contexts, schema, stats]

::: tabctx.cache.manager.ContextCacheManager
    options:
      members: [put, get, delete, contexts, stats]

## Values

::: tabctx.types.PredictOutcome

::: tabctx.types.ContextInfo

::: tabctx.types.EngineStats

## Errors

::: tabctx.errors
    options:
      show_root_heading: false
      members_order: source
