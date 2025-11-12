from __future__ import annotations

import pickle
from pathlib import Path

from config import DEFAULT_DATASET_USAGES
from optimizer import MultiDatasetHyperparameterOptimization
from visualizer import plot_comparative_convergence, plot_convergence


def load_and_visualize_results() -> None:
    """Load saved .pkl files and regenerate visualizations."""
    
    print("="*80)
    print("Loading and visualizing saved results...")
    print("="*80 + "\n")
    
    for dataset_usage in DEFAULT_DATASET_USAGES:
        pkl_path = f"results/usage_{dataset_usage}/optimization_results.pkl"
        
        if not Path(pkl_path).exists():
            print(f"Warning: {pkl_path} not found, skipping...")
            continue
        
        print(f"\nProcessing results for dataset usage {dataset_usage}%...")
        
        # Load the saved results
        with open(pkl_path, "rb") as f:
            saved_data = pickle.load(f)
        
        # Reconstruct the optimizer object with loaded data
        loaded_optimizer = MultiDatasetHyperparameterOptimization(
            openml_ids=saved_data["config"]["openml_ids"],
            ml_models=saved_data["config"]["ml_models"],
            sampling_methods=saved_data["config"]["sampling_methods"],
            dataset_usage_percent=saved_data["config"]["dataset_usage_percent"],
        )
        
        # Restore the results
        loaded_optimizer.results = saved_data["results"]
        loaded_optimizer.trial_results = saved_data["trial_results"]
        loaded_optimizer.datasets = saved_data["datasets"]
        if saved_data["complete_results"] is not None:
            loaded_optimizer.complete_results = saved_data["complete_results"]
        
        # Generate plots from loaded data
        print(f"Regenerating plots from saved data...")
        for metric in ("auc", "accuracy"):
            plot_convergence(
                loaded_optimizer,
                save_path=f"results/usage_{dataset_usage}/reloaded_convergence_plot_{metric}.png",
                metric=metric,
            )
            plot_comparative_convergence(
                loaded_optimizer,
                save_path=f"results/usage_{dataset_usage}/reloaded_comparative_convergence_{metric}.png",
                metric=metric,
            )
        
        print(f"Plots regenerated and saved with 'reloaded_' prefix")
        
        # Display summary statistics
        print(f"\nSummary for dataset usage {dataset_usage}%:")
        print(f"Number of datasets: {len(loaded_optimizer.datasets)}")
        print(f"Models evaluated: {list(loaded_optimizer.ml_models.keys())}")
        print(f"Sampling methods: {loaded_optimizer.sampling_methods}")
        
        print("\nBest results per model and method:")
        for key, result in loaded_optimizer.results.items():
            model_name = result.get("model_name", "unknown")
            method = result.get("sampling_method", "unknown")
            best_scores = result.get("best_scores", {})
            print(f"  {model_name.upper()} ({method}): AUC={best_scores.get('auc', 0):.4f}, Accuracy={best_scores.get('accuracy', 0):.4f}")
    
    print("\n" + "="*80)
    print("All results loaded and visualized successfully!")
    print("="*80)


if __name__ == "__main__":
    load_and_visualize_results()
