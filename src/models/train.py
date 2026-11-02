"""Model training, hyperparameter tuning with nested child runs, MLflow tracking, and model registration."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

# Ensure matplotlib uses a writable cache directory
mpl_dir = Path(tempfile.gettempdir()) / "mpl_cache"
mpl_dir.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config.config import AppConfig, config
from src.models.evaluate import (
    best_threshold_row,
    calculate_comprehensive_metrics,
    compute_precision_at_recalls,
    cost_ratio_sensitivity,
    generate_classification_report_file,
    plot_business_cost_curve,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_precision_recall_vs_threshold,
    plot_roc_curve,
    threshold_table,
)
from src.models.mlflow_utils import (
    get_git_commit_hash,
    register_production_model,
    setup_mlflow,
)

CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]
TARGET = "default"
SPLIT_KEY = "issue_d"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-input columns and cast credit history to whole years."""
    prepared = df.drop(columns=[SPLIT_KEY, "addr_state"], errors="ignore").copy()
    if "credit_hist_years" in prepared:
        prepared["credit_hist_years"] = prepared["credit_hist_years"].floordiv(1).astype("int64")
    return prepared


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split feature dataframe and binary target series."""
    return df.drop(columns=TARGET), df[TARGET]


def build_preprocessor() -> ColumnTransformer:
    """Build the feature preprocessing ColumnTransformer fitted on training period only."""
    return ColumnTransformer(
        transformers=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            )
        ],
        remainder="passthrough",
    )


def build_production_pipeline(raw_estimator: Any) -> Pipeline:
    """Build the end-to-end production pipeline accepting prepared feature columns."""
    preprocessor = build_preprocessor()
    estimator_steps = (
        raw_estimator.steps
        if isinstance(raw_estimator, Pipeline)
        else [("model", raw_estimator)]
    )
    return Pipeline([("preprocessor", preprocessor), *estimator_steps])


def log_grid_search_child_runs(grid_search: GridSearchCV, param_prefix: str = "") -> None:
    """Log individual hyperparameter combinations as nested child runs for comparison in the MLflow UI."""
    results = grid_search.cv_results_
    n_candidates = len(results["params"])

    for i in range(n_candidates):
        params = results["params"][i]
        with mlflow.start_run(nested=True, run_name=f"trial_{i+1:03d}"):
            # Flatten or format param names if prefixed with estimator steps
            clean_params = {
                (f"{param_prefix}{k.replace('model__', '')}"): (
                    str(v) if v is not None else "None"
                )
                for k, v in params.items()
            }
            mlflow.log_params(clean_params)

            metrics = {
                "mean_cv_roc_auc": float(results["mean_test_score"][i]),
                "std_cv_roc_auc": float(results["std_test_score"][i]),
            }
            if "mean_train_score" in results:
                metrics["mean_train_roc_auc"] = float(results["mean_train_score"][i])
                metrics["train_cv_gap"] = float(
                    results["mean_train_score"][i] - results["mean_test_score"][i]
                )
            mlflow.log_metrics(metrics)


def train_and_track_model_candidate(
    model_name: str,
    algorithm: str,
    grid_search: GridSearchCV,
    X_train_enc: np.ndarray | pd.DataFrame,
    y_train: pd.Series,
    X_valid_enc: np.ndarray | pd.DataFrame,
    y_valid: pd.Series,
    X_train_raw: pd.DataFrame,
    feature_names: list[str],
    train_period: tuple[str, str],
    valid_period: tuple[str, str],
    test_sample_count: int,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Train estimator via GridSearchCV, track nested trials, evaluate validation business metrics, and log artifacts."""
    active_cfg = cfg or config
    cost_ratio = active_cfg.business.cost_ratio
    min_recall = active_cfg.business.min_recall

    with mlflow.start_run(run_name=model_name) as parent_run:
        run_id = parent_run.info.run_id

        # 1. Fit grid search on training data
        grid_search.fit(X_train_enc, y_train)
        best_estimator = grid_search.best_estimator_

        # 2. Log nested child runs for hyperparameter trials
        log_grid_search_child_runs(grid_search)

        # 3. Model Information & Tags
        git_hash = get_git_commit_hash()
        mlflow.set_tags(
            {
                "project": "lending_club",
                "problem_type": "binary_classification",
                "task": "default_prediction",
                "dataset": "lending_club",
                "model_name": model_name,
                "algorithm": algorithm,
                "git_commit": git_hash or "unknown",
            }
        )

        # 4. Parameters
        model_info_params = {
            "model_name": model_name,
            "algorithm": algorithm,
            "model_type": type(best_estimator).__name__,
            "random_state": active_cfg.data.random_state,
            "training_period_start": train_period[0],
            "training_period_end": train_period[1],
            "validation_period_start": valid_period[0],
            "validation_period_end": valid_period[1],
            "feature_count": len(feature_names),
            "train_samples": len(y_train),
            "valid_samples": len(y_valid),
            "test_samples": test_sample_count,
            "cost_ratio_fn_to_fp": cost_ratio,
            "min_recall_floor": min_recall,
        }
        mlflow.log_params(model_info_params)

        # Log best hyperparameters
        best_params_clean = {
            f"best_{k.replace('model__', '')}": (str(v) if v is not None else "None")
            for k, v in grid_search.best_params_.items()
        }
        mlflow.log_params(best_params_clean)

        # Specific model hyperparameters
        if "XGB" in algorithm or isinstance(best_estimator, XGBClassifier):
            xgb_est = (
                best_estimator.named_steps["model"]
                if isinstance(best_estimator, Pipeline)
                else best_estimator
            )
            xgb_params = {
                "n_estimators": getattr(xgb_est, "n_estimators", 100),
                "max_depth": getattr(xgb_est, "max_depth", 3),
                "learning_rate": getattr(xgb_est, "learning_rate", 0.1),
                "min_child_weight": getattr(xgb_est, "min_child_weight", 1),
                "subsample": getattr(xgb_est, "subsample", 0.8),
                "colsample_bytree": getattr(xgb_est, "colsample_bytree", 0.8),
                "reg_alpha": getattr(xgb_est, "reg_alpha", 0),
                "reg_lambda": getattr(xgb_est, "reg_lambda", 1),
                "gamma": getattr(xgb_est, "gamma", 0),
                "objective": getattr(xgb_est, "objective", "binary:logistic"),
                "eval_metric": getattr(xgb_est, "eval_metric", "auc"),
                "tree_method": getattr(xgb_est, "tree_method", "hist"),
            }
            mlflow.log_params(xgb_params)
        elif "Logistic" in algorithm:
            lr_est = (
                best_estimator.named_steps["model"]
                if isinstance(best_estimator, Pipeline)
                else best_estimator
            )
            lr_params = {
                "C": getattr(lr_est, "C", 1.0),
                "class_weight": str(getattr(lr_est, "class_weight", None)),
                "max_iter": getattr(lr_est, "max_iter", 2000),
                "solver": getattr(lr_est, "solver", "lbfgs"),
            }
            mlflow.log_params(lr_params)

        # 5. Cross-Validation Metrics
        best_idx = grid_search.best_index_
        mean_cv_roc_auc = float(grid_search.best_score_)
        std_cv_roc_auc = float(grid_search.cv_results_["std_test_score"][best_idx])
        train_roc_auc = float(grid_search.cv_results_["mean_train_score"][best_idx])
        train_valid_gap = float(train_roc_auc - mean_cv_roc_auc)

        cv_metrics = {
            "cv_best_roc_auc": mean_cv_roc_auc,
            "cv_mean_roc_auc": mean_cv_roc_auc,
            "cv_std_roc_auc": std_cv_roc_auc,
            "train_roc_auc": train_roc_auc,
            "train_validation_gap": train_valid_gap,
        }
        mlflow.log_metrics(cv_metrics)

        # 6. Validation Performance & Business Threshold Selection
        valid_proba = best_estimator.predict_proba(X_valid_enc)[:, 1]
        thresholds_grid = np.round(
            np.arange(
                active_cfg.business.threshold_start,
                active_cfg.business.threshold_end,
                active_cfg.business.threshold_step,
            ),
            2,
        )
        val_threshold_df = threshold_table(
            y_valid, valid_proba, thresholds=thresholds_grid, cost_ratio=cost_ratio
        )
        best_val_row = best_threshold_row(val_threshold_df, min_recall=min_recall)
        selected_threshold = float(best_val_row["Threshold"])

        val_metrics = calculate_comprehensive_metrics(
            y_valid,
            valid_proba,
            threshold=selected_threshold,
            cost_ratio=cost_ratio,
            prefix="valid",
        )
        recall_target_precisions = compute_precision_at_recalls(
            y_valid, valid_proba, active_cfg.business.recall_levels
        )
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(
            {f"valid_{k}": v for k, v in recall_target_precisions.items()}
        )

        # 7. Build and Fit Production Pipeline (raw features -> prediction)
        production_pipeline = build_production_pipeline(clone(best_estimator))
        production_pipeline.fit(X_train_raw, y_train)

        # 8. Artifacts Generation and Logging
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # A. CV Results CSV
            cv_csv_path = tmp_path / "cv_results.csv"
            pd.DataFrame(grid_search.cv_results_).to_csv(cv_csv_path, index=False)
            mlflow.log_artifact(str(cv_csv_path), "hyperparameters")

            # B. Threshold Comparison CSV
            thresh_csv_path = tmp_path / "threshold_comparison.csv"
            val_threshold_df.to_csv(thresh_csv_path, index=False)
            mlflow.log_artifact(str(thresh_csv_path), "business_metrics")

            # C. Cost Ratio Sensitivity CSV
            sensitivity_df = cost_ratio_sensitivity(
                y_valid,
                valid_proba,
                cost_ratios=active_cfg.business.sensitivity_ratios,
            )
            sens_csv_path = tmp_path / "cost_ratio_sensitivity.csv"
            sensitivity_df.to_csv(sens_csv_path, index=False)
            mlflow.log_artifact(str(sens_csv_path), "business_metrics")

            # D. Curves
            roc_path = tmp_path / "roc_curve.png"
            plot_roc_curve(y_valid, valid_proba, model_name, roc_path)
            mlflow.log_artifact(str(roc_path), "plots")

            pr_path = tmp_path / "precision_recall_curve.png"
            plot_precision_recall_curve(
                y_valid, valid_proba, model_name, pr_path
            )
            mlflow.log_artifact(str(pr_path), "plots")

            val_preds = (valid_proba >= selected_threshold).astype(int)
            cm_path = tmp_path / "confusion_matrix.png"
            plot_confusion_matrix(y_valid, val_preds, model_name, cm_path)
            mlflow.log_artifact(str(cm_path), "plots")

            cost_curve_path = tmp_path / "business_cost_curve.png"
            plot_business_cost_curve(
                val_threshold_df,
                cost_ratio=cost_ratio,
                approve_all_cost=float(val_metrics["valid_approve_all_cost"]),
                selected_threshold=selected_threshold,
                output_path=cost_curve_path,
            )
            mlflow.log_artifact(str(cost_curve_path), "plots")

            pr_thresh_path = tmp_path / "precision_recall_vs_threshold.png"
            plot_precision_recall_vs_threshold(
                val_threshold_df,
                selected_threshold=selected_threshold,
                min_recall=min_recall,
                output_path=pr_thresh_path,
            )
            mlflow.log_artifact(str(pr_thresh_path), "plots")

            # E. Classification Report Text
            cls_rep_path = tmp_path / "classification_report.txt"
            generate_classification_report_file(y_valid, val_preds, cls_rep_path)
            mlflow.log_artifact(str(cls_rep_path), "evaluation")

            # F. Model Config, Feature Info, and Metadata JSONs
            feature_info_path = tmp_path / "feature_information.json"
            feature_info_path.write_text(
                json.dumps(
                    {
                        "feature_names": feature_names,
                        "feature_count": len(feature_names),
                        "input_columns": list(X_train_raw.columns),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(feature_info_path), "metadata")

            model_config_path = tmp_path / "model_config.json"
            model_config_path.write_text(
                json.dumps(
                    {
                        "model_name": model_name,
                        "algorithm": algorithm,
                        "best_params": grid_search.best_params_,
                        "threshold": selected_threshold,
                        "cost_ratio": cost_ratio,
                        "min_recall": min_recall,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(model_config_path), "metadata")

            eval_rep_path = tmp_path / "model_evaluation_report.json"
            eval_rep_path.write_text(
                json.dumps(
                    {
                        "model_name": model_name,
                        "cv_metrics": cv_metrics,
                        "validation_metrics": val_metrics,
                        "precision_at_recalls": recall_target_precisions,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(eval_rep_path), "evaluation")

            metadata_json_path = tmp_path / "model_metadata.json"
            metadata_dict = {
                "model_version": "1.0.0",
                "model_name": model_name,
                "threshold": selected_threshold,
                "cost_ratio": cost_ratio,
                "input_columns": list(X_train_raw.columns),
                "train_period": [train_period[0], train_period[1]],
                "validation_metrics": {
                    "roc_auc": val_metrics["valid_roc_auc"],
                    "pr_auc": val_metrics["valid_pr_auc"],
                    "precision_1": val_metrics["valid_precision_1"],
                    "recall_1": val_metrics["valid_recall_1"],
                    "f1_1": val_metrics["valid_f1_1"],
                    "cost": val_metrics["valid_cost"],
                    "tn": val_metrics["valid_tn"],
                    "fp": val_metrics["valid_fp"],
                    "fn": val_metrics["valid_fn"],
                    "tp": val_metrics["valid_tp"],
                },
                "versions": {
                    "scikit-learn": sklearn.__version__,
                    "xgboost": xgboost.__version__,
                    "pandas": pd.__version__,
                },
            }
            metadata_json_path.write_text(
                json.dumps(metadata_dict, indent=2), encoding="utf-8"
            )
            mlflow.log_artifact(str(metadata_json_path), "metadata")

        # 9. Log Production Scikit-learn Pipeline
        mlflow.sklearn.log_model(
            sk_model=production_pipeline,
            artifact_path="model",
            serialization_format="cloudpickle",
            registered_model_name=None,  # Registration handled separately with proper aliases
        )

        return {
            "model_name": model_name,
            "algorithm": algorithm,
            "run_id": run_id,
            "best_estimator": best_estimator,
            "production_pipeline": production_pipeline,
            "selected_threshold": selected_threshold,
            "valid_cost": float(val_metrics["valid_cost"]),
            "valid_metrics": val_metrics,
            "val_threshold_df": val_threshold_df,
            "input_columns": list(X_train_raw.columns),
            "train_period": train_period,
            "grid_search": grid_search,
        }


def run_complete_experimentation_workflow(
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Run complete experimentation: train 3 models, select Champion on validation data, evaluate Test once, register aliases."""
    active_cfg = cfg or config
    setup_mlflow(active_cfg)

    # 1. Load Data
    data_dir = active_cfg.data.data_dir
    train_df = pd.read_parquet(data_dir / "train.parquet")
    valid_df = pd.read_parquet(data_dir / "valid.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    # Time ordering
    train_sorted = train_df.sort_values(SPLIT_KEY, kind="stable")
    train_period = (str(train_sorted[SPLIT_KEY].min()), str(train_sorted[SPLIT_KEY].max()))
    valid_period = (str(valid_df[SPLIT_KEY].min()), str(valid_df[SPLIT_KEY].max()))

    X_train, y_train = split_xy(prepare(train_sorted))
    X_valid, y_valid = split_xy(prepare(valid_df))
    X_test, y_test = split_xy(prepare(test_df))

    input_columns = list(X_train.columns)
    X_train_raw = X_train

    # Encoder
    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_valid_enc = preprocessor.transform(X_valid)
    X_test_enc = preprocessor.transform(X_test)
    feature_names = list(X_train.columns)

    cv = TimeSeriesSplit(n_splits=3)
    random_state = active_cfg.data.random_state

    # ----------------- Model 1: Baseline XGBoost -----------------
    print("Training Candidate 1: XGBoost...", flush=True)
    xgb_search = GridSearchCV(
        estimator=XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        ),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.1],
            "min_child_weight": [1, 5],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=True,
    )
    xgb_res = train_and_track_model_candidate(
        model_name="XGBoost",
        algorithm="XGBoost (Hist)",
        grid_search=xgb_search,
        X_train_enc=X_train_enc,
        y_train=y_train,
        X_valid_enc=X_valid_enc,
        y_valid=y_valid,
        X_train_raw=X_train_raw,
        feature_names=feature_names,
        train_period=train_period,
        valid_period=valid_period,
        test_sample_count=len(y_test),
        cfg=active_cfg,
    )

    # ----------------- Model 2: Logistic Regression -----------------
    print("Training Candidate 2: Logistic Regression...", flush=True)
    lr_search = GridSearchCV(
        estimator=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        ),
        param_grid={
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__class_weight": [None, "balanced"],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    lr_res = train_and_track_model_candidate(
        model_name="Logistic Regression",
        algorithm="Logistic Regression",
        grid_search=lr_search,
        X_train_enc=X_train_enc,
        y_train=y_train,
        X_valid_enc=X_valid_enc,
        y_valid=y_valid,
        X_train_raw=X_train_raw,
        feature_names=feature_names,
        train_period=train_period,
        valid_period=valid_period,
        test_sample_count=len(y_test),
        cfg=active_cfg,
    )

    # ----------------- Model 3: XGBoost + Regularization -----------------
    print("Training Candidate 3: XGBoost + Regularization...", flush=True)
    reg_search = GridSearchCV(
        estimator=XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            **xgb_search.best_params_,
        ),
        param_grid={
            "reg_lambda": [1, 5, 10, 50],
            "reg_alpha": [0, 1, 5],
            "gamma": [0, 0.5, 1],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=True,
    )
    reg_res = train_and_track_model_candidate(
        model_name="XGBoost + regularization",
        algorithm="XGBoost Regularized (L1/L2/Gamma)",
        grid_search=reg_search,
        X_train_enc=X_train_enc,
        y_train=y_train,
        X_valid_enc=X_valid_enc,
        y_valid=y_valid,
        X_train_raw=X_train_raw,
        feature_names=feature_names,
        train_period=train_period,
        valid_period=valid_period,
        test_sample_count=len(y_test),
        cfg=active_cfg,
    )

    candidates = [xgb_res, reg_res, lr_res]

    # ----------------- Model Selection (Validation Data Only) -----------------
    # Minimize expected cost on validation set among eligible thresholds satisfying min_recall
    candidates_by_cost = sorted(candidates, key=lambda c: c["valid_cost"])
    champion_candidate = candidates_by_cost[0]
    challenger1_candidate = candidates_by_cost[1]
    challenger2_candidate = candidates_by_cost[2]

    print(f"\nModel Selection Result based on Validation Expected Cost:", flush=True)
    print(f"  Champion (Lowest Cost): {champion_candidate['model_name']} (Cost: {champion_candidate['valid_cost']:.4f})", flush=True)
    print(f"  Challenger 1: {challenger1_candidate['model_name']} (Cost: {challenger1_candidate['valid_cost']:.4f})", flush=True)
    print(f"  Challenger 2: {challenger2_candidate['model_name']} (Cost: {challenger2_candidate['valid_cost']:.4f})", flush=True)

    # ----------------- Unbiased Test Evaluation on Champion Only -----------------
    # Test set is evaluated exactly ONCE for the selected Champion at frozen threshold
    frozen_threshold = champion_candidate["selected_threshold"]
    test_proba = champion_candidate["best_estimator"].predict_proba(X_test_enc)[:, 1]
    test_metrics = calculate_comprehensive_metrics(
        y_test,
        test_proba,
        threshold=frozen_threshold,
        cost_ratio=active_cfg.business.cost_ratio,
        prefix="test",
    )

    # Log test metrics into the Champion's MLflow run
    with mlflow.start_run(run_id=champion_candidate["run_id"]):
        mlflow.log_metrics(test_metrics)
        mlflow.set_tag("model_role", "champion")

        # Test evaluation report
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_rep_path = tmp_path / "test_evaluation_report.json"
            test_rep_path.write_text(
                json.dumps(
                    {
                        "model_name": champion_candidate["model_name"],
                        "frozen_threshold": frozen_threshold,
                        "test_metrics": test_metrics,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(test_rep_path), "evaluation")

    # Set challenger tags on their runs
    with mlflow.start_run(run_id=challenger1_candidate["run_id"]):
        mlflow.set_tag("model_role", "challenger1")

    with mlflow.start_run(run_id=challenger2_candidate["run_id"]):
        mlflow.set_tag("model_role", "challenger2")

    # ----------------- Model Registry & Alias Management -----------------
    reg_name = active_cfg.mlflow.registered_model_name
    print(f"\nRegistering models to MLflow Model Registry: '{reg_name}'...", flush=True)

    champion_version = register_production_model(
        run_id=champion_candidate["run_id"],
        artifact_path="model",
        registered_model_name=reg_name,
        alias="champion",
        description=f"Champion Model: {champion_candidate['model_name']} selected on validation cost.",
        tags={"model_role": "champion", "algorithm": champion_candidate["algorithm"]},
    )

    challenger1_version = register_production_model(
        run_id=challenger1_candidate["run_id"],
        artifact_path="model",
        registered_model_name=reg_name,
        alias="challenger1",
        description=f"Challenger 1 Model: {challenger1_candidate['model_name']}",
        tags={"model_role": "challenger1", "algorithm": challenger1_candidate["algorithm"]},
    )

    challenger2_version = register_production_model(
        run_id=challenger2_candidate["run_id"],
        artifact_path="model",
        registered_model_name=reg_name,
        alias="challenger2",
        description=f"Challenger 2 Model: {challenger2_candidate['model_name']}",
        tags={"model_role": "challenger2", "algorithm": challenger2_candidate["algorithm"]},
    )

    print(f"Registered Champion: version {champion_version.version} (alias: @champion)", flush=True)
    print(f"Registered Challenger 1: version {challenger1_version.version} (alias: @challenger1)", flush=True)
    print(f"Registered Challenger 2: version {challenger2_version.version} (alias: @challenger2)", flush=True)

    # ----------------- Save Local Production Artifact -----------------
    models_dir = active_cfg.data.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = models_dir / "model.pkl"
    local_meta_path = models_dir / "model_metadata.json"

    # Save artifact dictionary as expected by existing notebook and backwards compatible with FastAPI
    production_artifact = {
        "pipeline": champion_candidate["production_pipeline"],
        "threshold": frozen_threshold,
        "model_name": champion_candidate["model_name"],
        "input_columns": champion_candidate["input_columns"],
        "cost_ratio": active_cfg.business.cost_ratio,
        "train_period": train_period,
        "versions": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
        },
    }
    joblib.dump(production_artifact, local_model_path)

    # Save model_metadata.json
    metadata_json = {
        "model_version": champion_version.version,
        "model_name": champion_candidate["model_name"],
        "run_id": champion_candidate["run_id"],
        "threshold": frozen_threshold,
        "cost_ratio": active_cfg.business.cost_ratio,
        "input_columns": champion_candidate["input_columns"],
        "train_period": [train_period[0], train_period[1]],
        "validation_metrics": champion_candidate["valid_metrics"],
        "test_metrics": test_metrics,
        "versions": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
        },
    }
    local_meta_path.write_text(json.dumps(metadata_json, indent=2), encoding="utf-8")
    print(f"\nLocal production artifact saved at {local_model_path} and {local_meta_path}", flush=True)

    return {
        "champion": champion_candidate,
        "challenger1": challenger1_candidate,
        "challenger2": challenger2_candidate,
        "champion_version": champion_version.version,
        "challenger1_version": challenger1_version.version,
        "challenger2_version": challenger2_version.version,
        "test_metrics": test_metrics,
    }
