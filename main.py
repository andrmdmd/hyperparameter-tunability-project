import os
import pandas as pd

from configs.datasets import load_dataset
from configs.model_spaces import MODEL_SPACES
from tuning.grid_search import run_grid_search
from tuning.random_search import run_random_search
from tuning.utils import get_model_pipeline

DATASETS = [
    1590,   # Adult
    1510,   #  Breast Cancer Wisconsin (Diagnostic)
]

METHODS = [
        #     "xgboost",
        #    "random_forest", 
           "logistic_regression"
           ]

SAMPLERS = {
    "grid": run_grid_search,
    # "random": run_random_search,
}

agg_scores = {method: {} for method in METHODS}
params_lookup = {method: {} for method in METHODS}

for ds in DATASETS:
    X_train, X_test, y_train, y_test = load_dataset(ds)
    for method in METHODS:
        for sampler_name, sampler_fn in SAMPLERS.items():
            model_info = MODEL_SPACES[method]
            cv_results = sampler_fn(
                model_info["model"],
                model_info["params"],
                X_train,
                y_train,
                metric="accuracy",
            )
            for p_dict, mean_score in zip(cv_results["params"], cv_results["mean_test_score"]):
                key = tuple(sorted(p_dict.items()))
                if key not in agg_scores[method]:
                    agg_scores[method][key] = []
                    params_lookup[method][key] = p_dict
                agg_scores[method][key].append(float(mean_score))

best_per_method = {}
for method in METHODS:
    best_key = None
    best_avg = -float("inf")
    for key, scores in agg_scores[method].items():
        avg = sum(scores) / len(scores)
        if avg > best_avg:
            best_avg = avg
            best_key = key
    if best_key is not None:
        best_per_method[method] = (params_lookup[method][best_key], best_avg)

final_results = []
for method, (best_params, avg_cv) in best_per_method.items():
    test_scores = []
    for ds in DATASETS:
        X_train, X_test, y_train, y_test = load_dataset(ds)
        pipeline = get_model_pipeline(MODEL_SPACES[method]["model"])
        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)
        test_scores.append(pipeline.score(X_test, y_test))
    final_results.append(
        {
            "method": method,
            "best_params": best_params,
            "avg_cv_score_across_datasets": avg_cv,
            "avg_test_score_across_datasets": sum(test_scores) / len(test_scores),
            "per_dataset_test_scores": test_scores,
        }
    )
    print(f"{method}: avg_cv={avg_cv:.4f} avg_test={sum(test_scores)/len(test_scores):.4f}")

os.makedirs("results", exist_ok=True)
pd.DataFrame(final_results).to_csv("results/tuning_results.csv", index=False)