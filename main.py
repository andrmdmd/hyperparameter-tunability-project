import os
import pandas as pd
from configs.datasets import load_dataset
from configs.model_spaces import MODEL_SPACES
from tuning.grid_search import run_grid_search
from tuning.random_search import run_random_search

DATASETS = ["iris", "wine", "cancer"]
METHODS = ["xgboost", "random_forest", "logistic_regression"]
SAMPLERS = {
    "grid": run_grid_search,
    "random": run_random_search,
}

results = []

for ds in DATASETS:
    X_train, X_test, y_train, y_test = load_dataset(ds)
    for method in METHODS:
        for sampler_name, sampler_fn in SAMPLERS.items():
            model_info = MODEL_SPACES[method]
            params, score = sampler_fn(
                model_info["model"],
                model_info["params"],
                X_train,
                y_train,
                metric="accuracy",
            )

            best_model = model_info["model"](**params)
            best_model.fit(X_train, y_train)
            test_score = best_model.score(X_test, y_test)

            results.append(
                {
                    "dataset": ds,
                    "method": method,
                    "sampler": sampler_name,
                    "best_params": params,
                    "best_score": score,
                    "test_score": test_score,
                }
            )
            print(f"[{ds}] {method}-{sampler_name}: {score:.3f} (test: {test_score:.3f})")

os.makedirs("results", exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv("results/tuning_results.csv", index=False)
