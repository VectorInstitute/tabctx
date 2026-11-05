#!/usr/bin/env python3
"""Prediction intervals from a tabctx server: fit California housing once,
then ask for each house's 5th, 50th and 95th percentile and check how often
the 90% interval holds the real price.

    TABCTX_BACKEND=tabicl python -m tabctx.serve      # in another shell
    python examples/prediction_intervals.py [--url URL] [--model MODEL]

Needs `pip install "tabctx[client]"` (scikit-learn downloads the data).
"""

import argparse

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from tabctx import TabctxClient, TabctxRegressor


def main(url: str, model: str | None) -> None:
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    X, y = X[:3000], y[:3000]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    reg = TabctxRegressor(model=model, client=TabctxClient(url)).fit(X_train, y_train)
    q = reg.predict_quantiles(X_test, quantiles=[0.05, 0.5, 0.95])
    low, median, high = q[:, 0], q[:, 1], q[:, 2]

    covered = float(((low <= y_test) & (y_test <= high)).mean())
    print(f"R^2 of the mean:            {reg.score(X_test, y_test):.3f}")
    print(f"90% interval holds price:   {covered:.1%} of {len(y_test)} houses")
    print(f"median interval width:      {float(np.median(high - low)):.2f} ($100k)")
    print(
        f"first house: {low[0]:.2f} .. {median[0]:.2f} .. {high[0]:.2f}"
        f" (actual {y_test.iloc[0]:.2f})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check prediction intervals.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", help="a model id; default: the server's default")
    args = parser.parse_args()
    main(args.url, args.model)
