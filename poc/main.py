from __future__ import annotations

from analyzer import analyze_tunability
from config import (
    DEFAULT_DATASET_USAGE,
    DEFAULT_SAMPLING_METHODS,
    ML_MODELS,
    OPENML_DATASETS,
    N_TRIALS,
)
from optimizer import MultiDatasetHyperparameterOptimization
from visualizer import plot_comparative_convergence, plot_convergence


def main() -> None:
    optimizer = MultiDatasetHyperparameterOptimization(
        openml_ids=OPENML_DATASETS,
        ml_models=ML_MODELS,
        sampling_methods=DEFAULT_SAMPLING_METHODS,
        dataset_usage_percent=DEFAULT_DATASET_USAGE,
    )

    print("Running optimization...")
    optimizer.run_complete_analysis(n_trials=N_TRIALS)

    print("Generating convergence plots...")
    for metric in ("auc", "accuracy"):
        plot_convergence(
            optimizer,
            save_path=f"convergence_plot_{metric}.png",
            metric=metric,
        )
        plot_comparative_convergence(
            optimizer,
            save_path=f"comparative_convergence_{metric}.png",
            metric=metric,
        )

    tunability_df = analyze_tunability(optimizer)
    print("\nTunability Analysis:")
    print(tunability_df)

    optimizer.save_results("optimization_results.pkl")
    print("Results saved successfully!")


if __name__ == "__main__":
    main()
