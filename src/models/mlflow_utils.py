"""MLflow tracking configuration, run management, artifact logging, and model registry with alias workflow."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from src.config.config import AppConfig, config


def get_git_commit_hash() -> str | None:
    """Safely obtain current git commit hash if running in a git repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def setup_mlflow(cfg: AppConfig | None = None) -> tuple[str, str]:
    """Configure MLflow tracking URI, artifact location, and ensure dedicated experiment exists."""
    active_cfg = cfg or config
    tracking_uri = active_cfg.mlflow.tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    exp_name = active_cfg.mlflow.experiment_name
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(exp_name)

    if experiment is None:
        # Create experiment with configured artifact location if provided
        artifact_loc = active_cfg.mlflow.artifact_location
        if artifact_loc and not artifact_loc.startswith("sqlite"):
            artifact_uri = Path(artifact_loc).resolve().as_uri()
            exp_id = client.create_experiment(name=exp_name, artifact_location=artifact_uri)
        else:
            exp_id = client.create_experiment(name=exp_name)
    else:
        exp_id = experiment.experiment_id

    mlflow.set_experiment(exp_name)
    return tracking_uri, exp_id


def get_mlflow_client() -> MlflowClient:
    """Return an initialized MlflowClient bound to the configured tracking URI."""
    setup_mlflow()
    return MlflowClient(tracking_uri=mlflow.get_tracking_uri())


def register_production_model(
    run_id: str,
    artifact_path: str = "model",
    registered_model_name: str | None = None,
    alias: str | None = None,
    description: str | None = None,
    tags: dict[str, str] | None = None,
) -> ModelVersion:
    """Register an MLflow model from a run, attach metadata tags, and optionally assign an alias."""
    active_reg_name = registered_model_name or config.mlflow.registered_model_name
    client = get_mlflow_client()

    # Ensure registered model entity exists
    try:
        client.get_registered_model(active_reg_name)
    except Exception:
        client.create_registered_model(
            name=active_reg_name,
            description="Lending Club Default Risk Prediction Model with complete serving pipeline.",
            tags={"project": "lending_club", "task": "default_prediction"},
        )

    model_uri = f"runs:/{run_id}/{artifact_path}"
    model_version = mlflow.register_model(model_uri=model_uri, name=active_reg_name)

    # Attach description and tags
    if description:
        client.update_model_version(
            name=active_reg_name,
            version=model_version.version,
            description=description,
        )

    all_tags = {
        "project": "lending_club",
        "problem_type": "binary_classification",
        "task": "default_prediction",
        "dataset": "lending_club",
        **(tags or {}),
    }
    for k, v in all_tags.items():
        client.set_model_version_tag(name=active_reg_name, version=model_version.version, key=k, value=str(v))

    # Assign alias if requested (e.g. champion, challenger1, challenger2)
    if alias:
        client.set_registered_model_alias(
            name=active_reg_name,
            alias=alias,
            version=model_version.version,
        )

    return client.get_model_version(name=active_reg_name, version=model_version.version)


def set_model_alias(alias: str, version: str | int, registered_model_name: str | None = None) -> None:
    """Safely assign or reassign an MLflow alias (e.g., 'champion', 'challenger1') to a specific version."""
    reg_name = registered_model_name or config.mlflow.registered_model_name
    client = get_mlflow_client()
    client.set_registered_model_alias(name=reg_name, alias=alias, version=str(version))


def get_model_by_alias(alias: str = "champion", registered_model_name: str | None = None) -> Any:
    """Load the full production pipeline from the MLflow model registry by alias."""
    reg_name = registered_model_name or config.mlflow.registered_model_name
    setup_mlflow()
    model_uri = f"models:/{reg_name}@{alias}"
    return mlflow.sklearn.load_model(model_uri)


def list_registered_versions(registered_model_name: str | None = None) -> list[dict[str, Any]]:
    """List all registered versions with their aliases, run ID, and tags for monitoring and rollback."""
    reg_name = registered_model_name or config.mlflow.registered_model_name
    client = get_mlflow_client()
    try:
        rm = client.get_registered_model(reg_name)
    except Exception:
        return []

    versions = client.search_model_versions(f"name='{reg_name}'")
    aliases_map = rm.aliases or {}  # alias -> version

    version_list = []
    for mv in versions:
        # Find which aliases point to this version
        assigned_aliases = [alias for alias, ver in aliases_map.items() if str(ver) == str(mv.version)]
        version_list.append(
            {
                "version": mv.version,
                "current_stage": mv.current_stage,
                "run_id": mv.run_id,
                "aliases": assigned_aliases,
                "description": mv.description,
                "tags": mv.tags,
                "creation_timestamp": mv.creation_timestamp,
            }
        )
    return sorted(version_list, key=lambda x: int(x["version"]), reverse=True)


def rollback_champion_to_version(target_version: str | int, registered_model_name: str | None = None) -> dict[str, Any]:
    """Roll back champion alias to a previous stable model version without retraining.

    Keeps the previous champion available as challenger1 if desired.
    """
    reg_name = registered_model_name or config.mlflow.registered_model_name
    client = get_mlflow_client()
    rm = client.get_registered_model(reg_name)
    previous_champion_version = (rm.aliases or {}).get("champion")

    # Set new champion
    client.set_registered_model_alias(name=reg_name, alias="champion", version=str(target_version))

    # Demote old champion to challenger1 if different
    if previous_champion_version and str(previous_champion_version) != str(target_version):
        client.set_registered_model_alias(name=reg_name, alias="challenger1", version=str(previous_champion_version))

    return {
        "status": "success",
        "current_champion": str(target_version),
        "demoted_to_challenger1": str(previous_champion_version) if previous_champion_version else None,
    }
