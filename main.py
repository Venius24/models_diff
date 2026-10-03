"""Small, reproducible comparison of three gradient boosting classifiers."""

import argparse
from time import perf_counter

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def main():
    parser = argparse.ArgumentParser(description="Compare three classifiers on synthetic data")
    parser.add_argument("--samples", type=int, default=50_000, help="number of generated rows")
    parser.add_argument("--estimators", type=int, default=100, help="trees per model")
    args = parser.parse_args()
    if args.samples < 100 or args.estimators < 1:
        parser.error("--samples must be >= 100 and --estimators must be >= 1")

    X, y = make_classification(
        n_samples=args.samples, n_features=20, n_informative=15, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2, stratify=y
    )

    models = {
        "LightGBM": LGBMClassifier(n_estimators=args.estimators, random_state=42, n_jobs=1, verbose=-1),
        "CatBoost": CatBoostClassifier(
            iterations=args.estimators, random_seed=42, thread_count=1,
            verbose=False, allow_writing_files=False
        ),
        "XGBoost": XGBClassifier(
            n_estimators=args.estimators, random_state=42, n_jobs=1,
            verbosity=0, eval_metric="logloss"
        ),
    }

    rows = []
    for name, model in models.items():
        start = perf_counter()
        model.fit(X_train, y_train)
        train_time = perf_counter() - start

        start = perf_counter()
        probabilities = model.predict_proba(X_test)[:, 1]
        inference_time = perf_counter() - start
        rows.append({
            "Model": name,
            "ROC-AUC": roc_auc_score(y_test, probabilities),
            "Train Time (s)": train_time,
            "Inference Time (s)": inference_time,
        })

    print(f"Samples: {args.samples}, trees per model: {args.estimators}, random_state: 42, threads: 1")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
