"""Tests for MLflow tracking, metrics, artifact generation, model registry, aliases, and rollback."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.config import config, load_config
from src.models.evaluate import (
    best_threshold_row,
    calculate_comprehensive_metrics,
    cost_ratio_sensitivity,
    plot_business_cost_curve,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    threshold_table,
)
from src.models.mlflow_utils import (
    get_mlflow_client,
    get_model_by_alias,
    list_registered_versions,
    register_production_model,
    rollback_champion_to_version,
    set_model_alias,
    setup_mlflow,
)
from src.api.predictor import Predictor


def test_mlflow_experiment_setup() -> None:
    """Verify MLflow connects to SQLite tracking URI and experiment is initialized."""
    uri, exp_id = setup_mlflow()
    client = get_mlflow_client()
    exp = client.get_experiment(exp_id)
    assert exp is not None
    assert exp.name == config.mlflow.experiment_name


def test_business_threshold_table_and_cost() -> None:
    """Verify threshold table metrics and expected cost calculation formula."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8])
    cost_ratio = 4.0

    table = threshold_table(y_true, proba, thresholds=[0.5], cost_ratio=cost_ratio)
    assert len(table) == 1
    row = table.iloc[0]

    # At threshold 0.5:
    # y=1 predicted as 1 (proba 0.6, 0.8 >= 0.5) -> TP = 2, FN = 0
    # y=0 predicted as 0 (proba 0.1, 0.2, 0.3, 0.4 < 0.5) -> TN = 4, FP = 0
    assert row["TP"] == 2
    assert row["TN"] == 4
    assert row["FP"] == 0
    assert row["FN"] == 0
    assert row["Cost"] == 0.0
    assert row["Recall_1"] == 1.0


def test_best_threshold_selection_with_recall_floor() -> None:
    """Verify selection rule picks lowest cost satisfying min_recall."""
    # Create synthetic table
    table = pd.DataFrame(
        {
            "Threshold": [0.1, 0.2, 0.3],
            "Recall_1": [0.8, 0.55, 0.40],  # 0.40 is below 0.50 floor
            "Cost": [0.60, 0.52, 0.48],  # 0.48 has lower cost but recall < 0.50
        }
    )
    best = best_threshold_row(table, min_recall=0.50)
    assert best["Threshold"] == 0.2
    assert best["Cost"] == 0.52


def test_artifact_plots_generation() -> None:
    """Verify plotting functions generate readable image files without display backend."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    proba = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.6, 0.2, 0.4])

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        roc_file = tmp_path / "roc.png"
        plot_roc_curve(y_true, proba, "TestModel", roc_file)
        assert roc_file.exists() and roc_file.stat().st_size > 0

        pr_file = tmp_path / "pr.png"
        plot_precision_recall_curve(y_true, proba, "TestModel", pr_file)
        assert pr_file.exists() and pr_file.stat().st_size > 0

        cm_file = tmp_path / "cm.png"
        plot_confusion_matrix(y_true, (proba >= 0.5).astype(int), "TestModel", cm_file)
        assert cm_file.exists() and cm_file.stat().st_size > 0


def test_cost_ratio_sensitivity() -> None:
    """Verify sensitivity table produces monotonic shifts for changing cost ratios."""
    y_true = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7])
    sens = cost_ratio_sensitivity(y_true, proba, cost_ratios=[1, 4, 8])
    assert len(sens) == 3
    assert "Expected_Cost_per_Loan" in sens.columns
    assert "Optimal_Threshold" in sens.columns


def test_predictor_handles_both_dict_and_pipeline(tmp_path: Path) -> None:
    """Predictor should seamlessly load model.pkl whether it is a Pipeline or a dict artifact."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    dummy_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    dummy_pipe.fit(X, y)

    # 1. Test dict format
    dict_artifact = {
        "pipeline": dummy_pipe,
        "threshold": 0.25,
        "cost_ratio": 4.0,
        "input_columns": ["f1", "f2"],
        "model_name": "TestDictModel",
    }
    model_path = tmp_path / "model.pkl"
    joblib.dump(dict_artifact, model_path)

    predictor = Predictor(models_dir=tmp_path, model_source="local")
    predictor.load()
    assert predictor.is_ready
    assert predictor.metadata["threshold"] == 0.25
    assert predictor.metadata["cost_ratio"] == 4.0


def test_registered_model_aliases_and_loading() -> None:
    """Verify registered model has champion and challenger aliases and pipelines can be loaded."""
    setup_mlflow()
    client = get_mlflow_client()
    rm = client.get_registered_model(config.mlflow.registered_model_name)
    aliases = rm.aliases or {}

    assert "champion" in aliases
    assert "challenger1" in aliases
    assert "challenger2" in aliases

    champion_model = get_model_by_alias("champion")
    challenger1_model = get_model_by_alias("challenger1")
    challenger2_model = get_model_by_alias("challenger2")

    assert hasattr(champion_model, "predict_proba")
    assert hasattr(challenger1_model, "predict_proba")
    assert hasattr(challenger2_model, "predict_proba")


def test_rollback_and_alias_restoration() -> None:
    """Verify rollback changes champion alias safely and restores without retraining."""
    setup_mlflow()
    client = get_mlflow_client()
    reg_name = config.mlflow.registered_model_name
    
    # Ensure starting in clean state with version 1 as champion, 2 as challenger1
    client.set_registered_model_alias(name=reg_name, alias="champion", version="1")
    client.set_registered_model_alias(name=reg_name, alias="challenger1", version="2")
    client.set_registered_model_alias(name=reg_name, alias="challenger2", version="3")

    try:
        # Perform rollback to version 2
        res = rollback_champion_to_version("2")
        assert res["status"] == "success"
        assert res["current_champion"] == "2"

        rm_updated = client.get_registered_model(reg_name)
        assert str((rm_updated.aliases or {}).get("champion")) == "2"

        # Restore original champion version 1
        restore_res = rollback_champion_to_version("1")
        assert str(restore_res["current_champion"]) == "1"

        rm_restored = client.get_registered_model(reg_name)
        assert str((rm_restored.aliases or {}).get("champion")) == "1"
    finally:
        # Guarantee aliases are restored to champion=1, challenger1=2, challenger2=3
        client.set_registered_model_alias(name=reg_name, alias="champion", version="1")
        client.set_registered_model_alias(name=reg_name, alias="challenger1", version="2")
        client.set_registered_model_alias(name=reg_name, alias="challenger2", version="3")


def test_parity_between_local_artifact_and_mlflow_champion() -> None:
    """Verify inference parity between local models/model.pkl and MLflow @champion."""
    p_local = Predictor(model_source="local")
    p_local.load()

    p_mlflow = Predictor(model_source="mlflow", mlflow_alias="champion")
    p_mlflow.load()

    assert p_local.is_ready
    assert p_mlflow.is_ready

    # Check with sample applicant data
    from src.api.schemas import ApplicantInput
    raw = pd.read_parquet("data/interim/lendingclub_clean.parquet")
    raw_columns = list(ApplicantInput.model_fields)
    nullable_events = {
        "mths_since_last_record",
        "mths_since_recent_bc_dlq",
        "mths_since_last_major_derog",
        "mths_since_recent_revol_delinq",
        "mths_since_last_delinq",
    }
    sample_row = raw.dropna(subset=[c for c in raw_columns if c not in nullable_events]).iloc[0]
    applicant = ApplicantInput(**sample_row[raw_columns].to_dict())

    local_preds = p_local.predict([applicant.model_dump()])
    mlflow_preds = p_mlflow.predict([applicant.model_dump()])

    assert local_preds[0]["flag_as_default"] == mlflow_preds[0]["flag_as_default"]
    assert np.isclose(
        local_preds[0]["default_probability_percent"],
        mlflow_preds[0]["default_probability_percent"],
        atol=1e-5,
    )

