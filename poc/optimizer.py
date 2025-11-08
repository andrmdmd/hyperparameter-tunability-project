from __future__ import annotations

import pickle
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from data_loader import DatasetDict, build_dataset_info, load_datasets

warnings.filterwarnings("ignore")


class MultiDatasetHyperparameterOptimization:
    def __init__(
        self,
        openml_ids: list[int],
        ml_models: Dict[str, Any],
        sampling_methods: list[str],
        dataset_usage_percent: float,
    ) -> None:
        self.openml_ids = openml_ids
        self.ml_models = ml_models
        self.sampling_methods = sampling_methods
        self.dataset_usage_percent = dataset_usage_percent

        self.datasets: DatasetDict = load_datasets(
            openml_ids, dataset_usage_percent
        )
        self.results: Dict[str, Dict[str, Any]] = {}
        self.trial_results: Dict[str, list[dict[str, Any]]] = {}

    def _apply_global_transform(self, value: Any, config: Dict[str, Any]) -> Any:
        transform = config.get("transform")
        if transform == "pow2":
            return float(np.power(2.0, value))
        return value

    def _build_dataset_params(
        self,
        model_name: str,
        params: Dict[str, Any],
        raw_params: Dict[str, Any],
        X: np.ndarray,
    ) -> Dict[str, Any]:
        param_configs = self.ml_models[model_name]["hyperparameters"]
        dataset_params: Dict[str, Any] = {}

        for param_name, config in param_configs.items():
            value = params[param_name]
            dataset_transform = config.get("dataset_transform")
            if dataset_transform == "max_features":
                features = X.shape[1]
                fraction = float(np.clip(value, 0.0, 1.0))
                candidate = int(np.ceil(fraction * features))
                candidate = max(1, min(features, candidate))
                dataset_params[param_name] = candidate
            elif dataset_transform == "min_samples_leaf_n_power":
                exponent_source = config.get("dataset_transform_source", "value")
                base_value = raw_params.get(param_name, value) if exponent_source == "raw" else value
                exponent = float(np.clip(base_value, 0.0, 1.0))
                n_samples = X.shape[0]
                candidate = int(np.round(np.power(n_samples, exponent)))
                candidate = max(1, min(n_samples, candidate))
                dataset_params[param_name] = candidate
            else:
                dataset_params[param_name] = value

        if model_name == "ranger":
            bootstrap = dataset_params.get("bootstrap", True)
            if not bootstrap:
                dataset_params["max_samples"] = None
            else:
                max_samples = dataset_params.get("max_samples")
                if max_samples is not None:
                    dataset_params["max_samples"] = float(np.clip(max_samples, 0.1, 1.0))

        return dataset_params

    def _sample_model_params(
        self, trial: optuna.Trial, model_name: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        param_configs = self.ml_models[model_name]["hyperparameters"]
        params: Dict[str, Any] = {}
        raw_params: Dict[str, Any] = {}

        for param_name, config in param_configs.items():
            if config["type"] == "categorical":
                value = trial.suggest_categorical(param_name, config["choices"])
            elif config["type"] == "float":
                value = trial.suggest_float(
                    param_name,
                    config["low"],
                    config["high"],
                    log=config.get("log", False),
                )
            elif config["type"] == "int":
                value = trial.suggest_int(
                    param_name,
                    config["low"],
                    config["high"],
                )
            else:  # pragma: no cover - guard
                raise ValueError(f"Unsupported hyperparameter type: {config['type']}")

            raw_params[param_name] = value
            params[param_name] = self._apply_global_transform(value, config)

        return params, raw_params

    def _create_model_pipeline(self, model_name: str, params: Dict[str, Any]) -> Pipeline:
        from sklearn.preprocessing import StandardScaler

        model_entry = self.ml_models[model_name]
        model_class = model_entry["class"]
        init_params = model_entry.get("init_params", {})
        model_kwargs = {**init_params, **params}

        if model_entry.get("scale"):
            return Pipeline([("scaler", StandardScaler()), ("model", model_class(**model_kwargs))])
        return Pipeline([("model", model_class(**model_kwargs))])

    def _build_grid_search_space(self, model_name: str) -> Dict[str, list[Any]]:
        search_space: Dict[str, list[Any]] = {}
        param_configs = self.ml_models[model_name]["hyperparameters"]

        for param_name, config in param_configs.items():
            if "grid_values" in config:
                search_space[param_name] = list(config["grid_values"])
                continue

            if config["type"] == "categorical":
                search_space[param_name] = list(config["choices"])
            elif config["type"] == "int":
                low, high = config["low"], config["high"]
                if high - low <= 20:
                    search_space[param_name] = list(range(low, high + 1))
                else:
                    steps = config.get("grid_steps", 5)
                    values = np.linspace(low, high, steps)
                    ints = sorted({int(round(v)) for v in values})
                    search_space[param_name] = [max(low, min(high, val)) for val in ints]
            elif config["type"] == "float":
                low, high = config["low"], config["high"]
                steps = config.get("grid_steps", 5)
                values = np.linspace(low, high, steps)
                search_space[param_name] = [float(v) for v in values]

        return search_space

    def _objective(self, trial: optuna.Trial, model_name: str, study_key: str) -> float:
        params, raw_params = self._sample_model_params(trial, model_name)

        dataset_results: Dict[str, float] = {}
        scores: list[float] = []

        try:
            for dataset_id, (X, y) in self.datasets.items():
                dataset_params = self._build_dataset_params(model_name, params, raw_params, X)
                pipeline = self._create_model_pipeline(model_name, dataset_params)
                cv_scores = cross_val_score(
                    pipeline, X, y, cv=5, scoring="accuracy", n_jobs=-1
                )
                dataset_score = float(np.mean(cv_scores))

                scores.append(dataset_score)
                dataset_results[str(dataset_id)] = dataset_score
                trial.set_user_attr(f"dataset_{dataset_id}", dataset_score)

            mean_score = float(np.mean(scores))

            trial_data = {
                "trial_number": trial.number,
                "params": params.copy(),
                "raw_params": raw_params.copy(),
                "mean_score": mean_score,
                "dataset_results": dataset_results.copy(),
                "dataset_usage_percent": self.dataset_usage_percent,
                "datetime": pd.Timestamp.now(),
            }

            if study_key not in self.trial_results:
                self.trial_results[study_key] = []
            self.trial_results[study_key].append(trial_data)

            trial.set_user_attr("dataset_results", dataset_results)
            trial.set_user_attr("mean_score", mean_score)
            trial.set_user_attr(
                "dataset_usage_percent", self.dataset_usage_percent
            )

            return mean_score
        except Exception as exc:  # pragma: no cover - safety net
            print(f"Error in trial {trial.number}: {exc}")
            trial_data = {
                "trial_number": trial.number,
                "params": params.copy(),
                "raw_params": raw_params.copy(),
                "mean_score": 0.0,
                "dataset_results": {
                    str(dataset_id): 0.0 for dataset_id in self.datasets.keys()
                },
                "dataset_usage_percent": self.dataset_usage_percent,
                "datetime": pd.Timestamp.now(),
                "error": str(exc),
            }
            if study_key not in self.trial_results:
                self.trial_results[study_key] = []
            self.trial_results[study_key].append(trial_data)
            return 0.0

    def optimize_model(
        self, model_name: str, sampling_method: str, n_trials: int
    ) -> Dict[str, Any]:
        print(
            f"Optimizing {model_name} using {sampling_method} sampling "
            f"(dataset usage: {self.dataset_usage_percent}%)..."
        )

        if sampling_method == "bayesian":
            sampler = optuna.samplers.TPESampler(seed=42)
        elif sampling_method == "random":
            sampler = optuna.samplers.RandomSampler(seed=42)
        elif sampling_method == "grid":
            search_space = self._build_grid_search_space(model_name)
            sampler = optuna.samplers.GridSampler(search_space, seed=42)
        else:  # pragma: no cover - caller guard
            raise ValueError(f"Unknown sampling method: {sampling_method}")

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study_key = f"{model_name}_{sampling_method}_{self.dataset_usage_percent}"

        study.optimize(
            lambda trial: self._objective(trial, model_name, study_key),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        trial_results = self.trial_results.get(study_key, [])
        result = {
            "study": study,
            "trial_results": trial_results,
            "best_params": study.best_params,
            "best_score": study.best_value,
            "model_name": model_name,
            "sampling_method": sampling_method,
            "dataset_usage_percent": self.dataset_usage_percent,
        }
        self.results[study_key] = result
        return result

    def run_complete_analysis(self, n_trials: int) -> Dict[str, Dict[str, Any]]:
        all_results: Dict[str, Dict[str, Any]] = {}
        for model_name in self.ml_models:
            model_results: Dict[str, Any] = {}
            for sampling_method in self.sampling_methods:
                model_results[sampling_method] = self.optimize_model(
                    model_name, sampling_method, n_trials
                )
            all_results[model_name] = model_results
        self.complete_results = all_results
        return all_results

    def get_dataset_info(self) -> pd.DataFrame:
        info = build_dataset_info(self.datasets, self.dataset_usage_percent)
        return pd.DataFrame(info)

    def save_results(self, filename: str) -> None:
        with open(filename, "wb") as file_handle:
            pickle.dump(
                {
                    "results": self.results,
                    "trial_results": self.trial_results,
                    "complete_results": getattr(self, "complete_results", None),
                    "datasets": self.datasets,
                    "dataset_info": self.get_dataset_info(),
                    "config": {
                        "openml_ids": self.openml_ids,
                        "ml_models": self.ml_models,
                        "sampling_methods": self.sampling_methods,
                        "dataset_usage_percent": self.dataset_usage_percent,
                    },
                },
                file_handle,
            )
