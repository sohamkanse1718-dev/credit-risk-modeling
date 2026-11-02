"""Model loading and inference supporting both local artifacts and MLflow Model Registry."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config.config import config
from src.pipeline.training_pipeline import prepare_model_inputs


class Predictor:
    """Load the trained production pipeline and its metadata from local disk or MLflow Registry."""

    def __init__(
        self,
        models_dir: Path | None = None,
        model_source: str | None = None,
        mlflow_alias: str | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        self.models_dir = models_dir or root / "models"
        self.model_source = (
            model_source or os.getenv("MODEL_SOURCE") or config.api.model_source
        ).lower()
        self.mlflow_alias = (
            mlflow_alias
            or os.getenv("MLFLOW_MODEL_ALIAS")
            or config.api.mlflow_model_alias
        )
        self.pipeline: Any | None = None
        self.metadata: dict[str, Any] | None = None
        self.error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None and self.metadata is not None

    def load(self) -> None:
        """Load and validate artifacts from local storage or MLflow Model Registry."""
        try:
            if self.model_source == "mlflow":
                self._load_from_mlflow()
            else:
                self._load_from_local()

            if not hasattr(self.pipeline, "predict_proba"):
                raise TypeError("Loaded model does not implement predict_proba.")

            for key in ("threshold", "cost_ratio", "model_version", "input_columns", "validation_metrics"):
                if key not in self.metadata:
                    raise ValueError(f"Model metadata is missing '{key}'.")

            self.error = None
        except Exception as exc:  # surfaced by /ready and converted to HTTP 503 by routes
            self.pipeline, self.metadata, self.error = None, None, str(exc)

    def _load_from_local(self) -> None:
        """Load from local models directory (model.pkl and optional model_metadata.json)."""
        model_path = self.models_dir / "model.pkl"
        metadata_path = self.models_dir / "model_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Local model not found at {model_path}")

        loaded_obj = joblib.load(model_path)

        # Backward compatibility: support both dict artifact and pure Pipeline
        if isinstance(loaded_obj, dict) and "pipeline" in loaded_obj:
            pipeline = loaded_obj["pipeline"]
            threshold = float(loaded_obj.get("threshold", 0.19))
            cost_ratio = float(loaded_obj.get("cost_ratio", 4.0))
            input_columns = loaded_obj.get("input_columns", [])
            model_name = loaded_obj.get("model_name", "LendingClubDefaultRiskModel")
        else:
            pipeline = loaded_obj
            threshold = 0.19
            cost_ratio = 4.0
            input_columns = []
            model_name = "LendingClubDefaultRiskModel"

        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            metadata = {
                "model_version": "1.0.0",
                "model_name": model_name,
                "threshold": threshold,
                "cost_ratio": cost_ratio,
                "input_columns": input_columns,
                "validation_metrics": {},
            }

        # Ensure required keys exist
        if "threshold" not in metadata:
            metadata["threshold"] = threshold
        if "cost_ratio" not in metadata:
            metadata["cost_ratio"] = cost_ratio
        if "input_columns" not in metadata or not metadata["input_columns"]:
            metadata["input_columns"] = input_columns
        if "model_version" not in metadata:
            metadata["model_version"] = "1.0.0"
        if "validation_metrics" not in metadata:
            metadata["validation_metrics"] = {}

        self.pipeline = pipeline
        self.metadata = metadata

    def _load_from_mlflow(self) -> None:
        """Load a registered model version from MLflow Model Registry via alias."""
        from src.models.mlflow_utils import get_mlflow_client, setup_mlflow
        import mlflow.sklearn

        setup_mlflow()
        client = get_mlflow_client()
        reg_name = config.mlflow.registered_model_name
        alias = self.mlflow_alias

        model_uri = f"models:/{reg_name}@{alias}"
        pipeline = mlflow.sklearn.load_model(model_uri)

        # Retrieve version metadata from registry
        rm = client.get_registered_model(reg_name)
        version_num = (rm.aliases or {}).get(alias)
        mv = client.get_model_version(reg_name, str(version_num)) if version_num else None

        # Load metadata from model run or fallback to local
        metadata_path = self.models_dir / "model_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            metadata = {
                "threshold": 0.19,
                "cost_ratio": 4.0,
                "validation_metrics": {},
                "input_columns": [],
            }

        if mv:
            metadata["model_version"] = str(mv.version)
            metadata["model_name"] = mv.tags.get("model_name", f"{reg_name}@{alias}")
            metadata["mlflow_alias"] = alias
            metadata["mlflow_run_id"] = mv.run_id

        self.pipeline = pipeline
        self.metadata = metadata

    def _require_ready(self) -> tuple[Any, dict[str, Any]]:
        if not self.is_ready:
            raise RuntimeError(self.error or "Model is not ready.")
        return self.pipeline, self.metadata  # type: ignore[return-value]

    @staticmethod
    def _risk(probability: float, threshold: float) -> tuple[str, str]:
        """Return a risk category and its displayed probability range."""
        midpoint_percent = threshold * 50
        threshold_percent = threshold * 100
        if probability >= threshold:
            return "high", f">= {threshold_percent:g}%"
        if probability >= threshold / 2:
            return "medium", f">= {midpoint_percent:g}% and < {threshold_percent:g}%"
        return "low", f"< {midpoint_percent:g}%"

    @staticmethod
    def risk_categories(threshold: float) -> dict[str, str]:
        """Return the policy used to turn probabilities into display categories."""
        midpoint_percent = threshold * 50
        threshold_percent = threshold * 100
        return {
            "low": f"< {midpoint_percent:g}%",
            "medium": f">= {midpoint_percent:g}% and < {threshold_percent:g}%",
            "high": f">= {threshold_percent:g}% (flagged for review)",
        }

    def _frame(self, applicants: Iterable[dict[str, Any]]) -> pd.DataFrame:
        _, metadata = self._require_ready()
        frame = pd.DataFrame(list(applicants))
        prepared = prepare_model_inputs(frame)
        expected = metadata["input_columns"]
        missing = sorted(set(expected) - set(prepared.columns))
        if missing:
            raise ValueError(f"Feature preparation did not create: {', '.join(missing)}")
        return prepared.loc[:, expected]

    @staticmethod
    def _display_feature_name(feature: str) -> str:
        """Make sklearn transformer feature names readable in API responses."""
        return feature.removeprefix("encoder__").removeprefix("remainder__")

    def _shap_contributions(self, pipeline: Any, frame: pd.DataFrame) -> list[list[dict[str, Any]]]:
        """Return the ten strongest feature contributions for this prediction.

        For tree models (XGBoost), computes native TreeSHAP in log-odds space.
        For linear models (Logistic Regression), computes linear coefficient contributions.
        """
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        transformed = preprocessor.transform(frame)
        feature_names = preprocessor.get_feature_names_out()

        # 1. TreeSHAP for XGBoost
        if hasattr(model, "get_booster"):
            contributions = model.get_booster().predict(xgb.DMatrix(transformed), pred_contribs=True)[:, :-1]
        # 2. Linear coefficient impact for Logistic Regression or other linear estimators
        elif hasattr(model, "coef_"):
            coef = model.coef_[0]
            if hasattr(transformed, "toarray"):
                transformed_dense = transformed.toarray()
            else:
                transformed_dense = np.asarray(transformed)
            contributions = transformed_dense * coef
        # 3. Fallback dummy contributions
        else:
            contributions = np.zeros((len(frame), len(feature_names)))

        explanations: list[list[dict[str, Any]]] = []
        for row in contributions:
            denominator = float(np.abs(row).sum()) or 1.0
            top_indices = np.argsort(np.abs(row))[::-1][:10]
            explanations.append(
                [
                    {
                        "feature": self._display_feature_name(str(feature_names[index])),
                        "shap_value": float(row[index]),
                        "impact_percent": float(abs(row[index]) / denominator * 100),
                        "effect": "increases_default_risk" if row[index] >= 0 else "reduces_default_risk",
                    }
                    for index in top_indices
                ]
            )
        return explanations

    def predict(self, applicants: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        pipeline, metadata = self._require_ready()
        frame = self._frame(applicants)
        probabilities = pipeline.predict_proba(frame)[:, 1]
        shap_values = self._shap_contributions(pipeline, frame)
        threshold = float(metadata["threshold"])
        return [
            {
                "default_probability_percent": round(float(probability) * 100, 2),
                "flag_as_default": bool(probability >= threshold),
                "risk_category": self._risk(float(probability), threshold)[0],
                "risk_range_percent": self._risk(float(probability), threshold)[1],
                "decision": "flagged_for_review" if probability >= threshold else "not_flagged",
                "threshold_percent": threshold * 100,
                "model_version": str(metadata["model_version"]),
                "shap_top_features": shap_values[index],
            }
            for index, probability in enumerate(np.asarray(probabilities))
        ]

    def model_info(self) -> dict[str, Any]:
        _, metadata = self._require_ready()
        return {**metadata, "risk_categories": self.risk_categories(float(metadata["threshold"]))}
