"""Serving invariants: readiness semantics and notebook/pipeline parity."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from src.api import main
from src.api.predictor import Predictor
from src.api.schemas import ApplicantInput
from src.pipeline.training_pipeline import prepare_model_inputs


def test_ready_returns_503_without_a_loaded_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Readiness is state-only; it must not load from disk or claim success."""
    unavailable = Predictor()
    unavailable.error = "model has not been loaded"
    unavailable.load = Mock()  # type: ignore[method-assign]
    monkeypatch.setattr(main, "predictor", unavailable)

    with pytest.raises(HTTPException) as error:
        main.ready()

    assert error.value.status_code == 503
    unavailable.load.assert_not_called()


def test_raw_api_features_match_processed_pipeline_probability() -> None:
    """The raw API path must produce the exact saved-pipeline notebook score."""
    metadata = json.loads(Path("models/model_metadata.json").read_text(encoding="utf-8"))
    raw = pd.read_parquet("data/interim/lendingclub_clean.parquet")
    raw_valid = raw[(raw["issue_d"] >= "2013-01-01") & (raw["issue_d"] < "2014-01-01")].reset_index(drop=True)
    processed_valid = pd.read_parquet("data/processed/valid.parquet")

    raw_columns = list(ApplicantInput.model_fields)
    nullable_events = {
        "mths_since_last_record",
        "mths_since_recent_bc_dlq",
        "mths_since_last_major_derog",
        "mths_since_recent_revol_delinq",
        "mths_since_last_delinq",
    }
    row_index = raw_valid.dropna(subset=[column for column in raw_columns if column not in nullable_events]).index[0]
    applicant = ApplicantInput(**raw_valid.loc[row_index, raw_columns].to_dict())

    predictor = Predictor()
    predictor.load()
    api_frame = predictor._frame([applicant.model_dump()])
    api_probability = predictor.pipeline.predict_proba(api_frame)[0, 1]
    notebook_probability = predictor.pipeline.predict_proba(
        prepare_model_inputs(processed_valid.iloc[[row_index]])
    )[0, 1]

    assert metadata["threshold"] == 0.19
    assert np.isclose(api_probability, notebook_probability, rtol=0, atol=1e-12)
