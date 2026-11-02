"""Configuration management for MLflow tracking, business logic, data paths, and serving."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MLflowConfig:
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "Lending-Club-Default-Risk-Experiments"
    artifact_location: str = "mlruns"
    registered_model_name: str = "LendingClubDefaultRiskModel"


@dataclass
class BusinessConfig:
    cost_ratio: float = 4.0
    min_recall: float = 0.50
    threshold_start: float = 0.01
    threshold_end: float = 1.00
    threshold_step: float = 0.01
    recall_levels: list[float] = field(default_factory=lambda: [0.30, 0.40, 0.50, 0.60])
    sensitivity_ratios: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 8])


@dataclass
class DataConfig:
    data_dir: Path = Path("data/processed")
    interim_dir: Path = Path("data/interim")
    models_dir: Path = Path("models")
    random_state: int = 42


@dataclass
class APIConfig:
    model_source: str = "local"
    mlflow_model_alias: str = "champion"


@dataclass
class AppConfig:
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    business: BusinessConfig = field(default_factory=BusinessConfig)
    data: DataConfig = field(default_factory=DataConfig)
    api: APIConfig = field(default_factory=APIConfig)


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """Load configuration from YAML file and apply environment variable overrides."""
    root = get_project_root()
    if config_path is None:
        config_path = root / "configs" / "config.yaml"
        if not config_path.exists() or config_path.stat().st_size == 0:
            example_path = root / "configs" / "config.example.yaml"
            if example_path.exists():
                config_path = example_path
    else:
        config_path = Path(config_path)

    data: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    # If data is empty after loading, fall back to config.example.yaml
    if not data:
        example_path = root / "configs" / "config.example.yaml"
        if example_path.exists():
            with open(example_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

    mlflow_data = data.get("mlflow", {})
    business_data = data.get("business", {})
    data_data = data.get("data", {})
    api_data = data.get("api", {})

    # Ensure tracking URI uses an absolute path if it is a local sqlite file
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", mlflow_data.get("tracking_uri", "sqlite:///mlflow.db"))
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////") and not ":" in tracking_uri[10:12]:
        # relative sqlite path -> resolve to absolute
        rel_db = tracking_uri.replace("sqlite:///", "")
        abs_db = (root / rel_db).resolve().as_posix()
        tracking_uri = f"sqlite:///{abs_db}"

    mlflow_cfg = MLflowConfig(
        tracking_uri=tracking_uri,
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", mlflow_data.get("experiment_name", "Lending-Club-Default-Risk-Experiments")),
        artifact_location=os.getenv("MLFLOW_ARTIFACT_LOCATION", str((root / mlflow_data.get("artifact_location", "mlruns")).resolve())),
        registered_model_name=os.getenv("MLFLOW_REGISTERED_MODEL_NAME", mlflow_data.get("registered_model_name", "LendingClubDefaultRiskModel")),
    )

    biz_cfg = BusinessConfig(
        cost_ratio=float(os.getenv("COST_RATIO", business_data.get("cost_ratio", 4.0))),
        min_recall=float(os.getenv("MIN_RECALL", business_data.get("min_recall", 0.50))),
        threshold_start=float(business_data.get("threshold_start", 0.01)),
        threshold_end=float(business_data.get("threshold_end", 1.00)),
        threshold_step=float(business_data.get("threshold_step", 0.01)),
        recall_levels=business_data.get("recall_levels", [0.30, 0.40, 0.50, 0.60]),
        sensitivity_ratios=business_data.get("sensitivity_ratios", [1, 2, 3, 4, 5, 8]),
    )

    data_cfg = DataConfig(
        data_dir=root / data_data.get("data_dir", "data/processed"),
        interim_dir=root / data_data.get("interim_dir", "data/interim"),
        models_dir=root / data_data.get("models_dir", "models"),
        random_state=int(data_data.get("random_state", 42)),
    )

    api_cfg = APIConfig(
        model_source=os.getenv("MODEL_SOURCE", api_data.get("model_source", "local")),
        mlflow_model_alias=os.getenv("MLFLOW_MODEL_ALIAS", api_data.get("mlflow_model_alias", "champion")),
    )

    return AppConfig(mlflow=mlflow_cfg, business=biz_cfg, data=data_cfg, api=api_cfg)


# Default global instance
config = load_config()
