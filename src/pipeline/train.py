"""CLI entry point for running the complete MLflow training and model registry workflow."""

import argparse
from pathlib import Path

from src.models.train import run_complete_experimentation_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete MLflow experiment tracking and model registry workflow.")
    parser.parse_args()
    print("Starting MLflow experiment tracking and model registration workflow...")
    results = run_complete_experimentation_workflow()
    champion = results["champion"]
    print("\n========================================================")
    print("Workflow completed successfully!")
    print(f"Champion Model: {champion['model_name']} (Version {results['champion_version']})")
    print(f"Validation Cost: {champion['valid_cost']:.4f} | Frozen Threshold: {champion['selected_threshold']:.2f}")
    print(f"Test ROC-AUC: {results['test_metrics']['test_roc_auc']:.4f} | Test PR-AUC: {results['test_metrics']['test_pr_auc']:.4f}")
    print(f"Test Expected Cost: {results['test_metrics']['test_cost']:.4f}")
    print("Registered model alias '@champion' points to this version.")
    print("========================================================\n")


if __name__ == "__main__":
    main()
