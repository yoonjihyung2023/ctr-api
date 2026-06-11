from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="CTR API",
    description="FastAPI demo for CTR prediction serving.",
    version="0.1.0",
)


class PredictRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="User identifier")
    item_id: Optional[str] = Field(default=None, description="Item identifier")
    features: Dict[str, Any] = Field(default_factory=dict, description="Input feature dictionary")


class PredictResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="CTR probability score in [0, 1]")
    model: str = "demo-ctr-model"


def clamp_score(score: float) -> float:
    """Clamp prediction score to a valid probability range [0.0, 1.0]."""
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.5
    return max(0.0, min(1.0, score))


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "ctr-api"}


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    return {
        "model": "demo-ctr-model",
        "task": "CTR prediction",
        "output": "score in [0, 1]",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    # Demo scoring logic.
    # In production, replace this with a loaded model prediction.
    raw_score = 0.5

    if request.user_id:
        raw_score += 0.05

    if request.item_id:
        raw_score += 0.05

    if request.features:
        raw_score += min(len(request.features) * 0.01, 0.2)

    score = clamp_score(raw_score)
    return PredictResponse(score=score)
