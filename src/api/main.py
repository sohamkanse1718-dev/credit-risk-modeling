"""FastAPI routes for the Lending Club default-risk model."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .predictor import Predictor
from .schemas import (
    ApplicantInput,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadyResponse,
    ServiceInfoResponse,
)

predictor = Predictor()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load once at application startup; readiness never reloads from disk."""
    predictor.load()
    yield


app = FastAPI(
    title="Lending Club Default Risk API",
    version="1.0.0",
    lifespan=lifespan,
)


def _unavailable(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=f"Model is unavailable: {exc}")


@app.get("/", response_model=ServiceInfoResponse)
def root() -> ServiceInfoResponse:
    return ServiceInfoResponse(
        service="Lending Club Default Risk Prediction API",
        message="Welcome. Submit applicant features to estimate default risk and view TreeSHAP feature drivers.",
        docs_url="/docs",
        health_url="/health",
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail=predictor.error or "Model is not loaded.")
    return ReadyResponse(ready=True, model_loaded=True, detail=None)


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        return ModelInfoResponse(**predictor.model_info())
    except Exception as exc:
        raise _unavailable(exc) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantInput) -> PredictionResponse:
    try:
        return PredictionResponse(**predictor.predict([applicant.model_dump()])[0])
    except Exception as exc:
        raise _unavailable(exc) from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    try:
        predictions = [PredictionResponse(**item) for item in predictor.predict(item.model_dump() for item in request.applicants)]
        return BatchPredictionResponse(predictions=predictions, count=len(predictions))
    except Exception as exc:
        raise _unavailable(exc) from exc
