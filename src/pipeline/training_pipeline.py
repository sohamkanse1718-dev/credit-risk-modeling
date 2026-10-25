"""Reproducible training for the Lending Club default model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBClassifier

from src.features import engineer_application_features

TARGET = "default"
SPLIT_KEY = "issue_d"
CATEGORICAL_COLUMNS = ["home_ownership", "purpose", "verification_status"]
THRESHOLD = 0.19
COST_RATIO = 4
MODEL_VERSION = "1.0.0"


def prepare_model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply serving-time feature rules and retain only estimator inputs."""
    prepared = engineer_application_features(frame)
    prepared = prepared.drop(columns=[TARGET, SPLIT_KEY, "addr_state"], errors="ignore")
    # The selected notebook model consumed whole credit-history years.
    if "credit_hist_years" in prepared:
        prepared["credit_hist_years"] = prepared["credit_hist_years"].floordiv(1).astype("int64")
    return prepared


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in frame:
        raise ValueError(f"Expected target column '{TARGET}'.")
    return prepare_model_inputs(frame), frame[TARGET]


def build_pipeline() -> Pipeline:
    """Build the exact production preprocessing and selected XGBoost model."""
    preprocessor = ColumnTransformer(
        [("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS)],
        remainder="passthrough",
    )
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=50,
        reg_alpha=0,
        gamma=0,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def threshold_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float | int]:
    """Compute the validation metrics recorded alongside the model artifact."""
    y = np.asarray(y_true)
    predicted = probabilities >= THRESHOLD
    tp = int((predicted & (y == 1)).sum())
    fp = int((predicted & (y == 0)).sum())
    fn = int((~predicted & (y == 1)).sum())
    tn = int((~predicted & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "precision_1": precision,
        "recall_1": recall,
        "f1_1": 2 * precision * recall / max(precision + recall, 1e-12),
        "cost": (COST_RATIO * fn + fp) / len(y),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def train(data_dir: Path, models_dir: Path) -> dict[str, Any]:
    """Fit from ``data/processed`` and write one pipeline plus JSON metadata."""
    train_frame = pd.read_parquet(data_dir / "train.parquet").sort_values(SPLIT_KEY, kind="stable")
    valid_frame = pd.read_parquet(data_dir / "valid.parquet")
    test_frame = pd.read_parquet(data_dir / "test.parquet")

    x_train, y_train = split_xy(train_frame)
    x_valid, y_valid = split_xy(valid_frame)
    x_test, y_test = split_xy(test_frame)
    if list(x_train.columns) != list(x_valid.columns) or list(x_train.columns) != list(x_test.columns):
        raise ValueError("Processed splits do not have identical model-input columns.")

    pipeline = build_pipeline().fit(x_train, y_train)
    validation_metrics = threshold_metrics(y_valid, pipeline.predict_proba(x_valid)[:, 1])
    test_metrics = threshold_metrics(y_test, pipeline.predict_proba(x_test)[:, 1])

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.pkl"
    metadata_path = models_dir / "model_metadata.json"
    joblib.dump(pipeline, model_path)
    metadata: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "model_name": "XGBoost + regularization",
        "threshold": THRESHOLD,
        "cost_ratio": COST_RATIO,
        "input_columns": list(x_train.columns),
        "train_period": [str(train_frame[SPLIT_KEY].min()), str(train_frame[SPLIT_KEY].max())],
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "pandas": pd.__version__},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
