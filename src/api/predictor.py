"""Model loading and inference; no HTTP concerns live in this module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.pipeline.training_pipeline import prepare_model_inputs


class Predictor:
    """Load the single saved sklearn pipeline and its adjacent metadata."""

    def __init__(self, models_dir: Path | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self.models_dir = models_dir or root / "models"
        self.pipeline: Any | None = None
        self.metadata: dict[str, Any] | None = None
        self.error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None and self.metadata is not None

    def load(self) -> None:
        """Load and validate both artifacts, retaining an actionable readiness error."""
        try:
            metadata_path = self.models_dir / "model_metadata.json"
            model_path = self.models_dir / "model.pkl"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in ("threshold", "cost_ratio", "model_version", "input_columns", "validation_metrics"):
                if key not in metadata:
                    raise ValueError(f"Model metadata is missing '{key}'.")
            pipeline = joblib.load(model_path)
            if not hasattr(pipeline, "predict_proba"):
                raise TypeError("Saved model does not implement predict_proba.")
            self.pipeline, self.metadata, self.error = pipeline, metadata, None
        except Exception as exc:  # surfaced by /ready and converted to HTTP 503 by routes
            self.pipeline, self.metadata, self.error = None, None, str(exc)

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
        # This shared function derives credit history, loan-to-income, every
        # missingness flag, ZIP aggregation, term and employment-length values.
        prepared = prepare_model_inputs(frame)
        expected = metadata["input_columns"]
        missing = sorted(set(expected) - set(prepared.columns))
        if missing:
            raise ValueError(f"Feature preparation did not create: {', '.join(missing)}")
        # Match the exact processed-feature schema used at training time.
        return prepared.loc[:, expected]

    @staticmethod
    def _display_feature_name(feature: str) -> str:
        """Make sklearn transformer feature names readable in API responses."""
        return feature.removeprefix("encoder__").removeprefix("remainder__")

    def _shap_contributions(self, pipeline: Any, frame: pd.DataFrame) -> list[list[dict[str, Any]]]:
        """Return the ten strongest native XGBoost TreeSHAP feature effects.

        SHAP values are in log-odds space. ``impact_percent`` is each feature's
        share of the total absolute SHAP impact for this prediction, which is a
        readable ranking rather than a probability contribution.
        """
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        transformed = preprocessor.transform(frame)
        feature_names = preprocessor.get_feature_names_out()
        # ``pred_contribs=True`` returns one additional final column: base value.
        contributions = model.get_booster().predict(xgb.DMatrix(transformed), pred_contribs=True)[:, :-1]
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
