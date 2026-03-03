from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os

app = FastAPI(title="ctr-api", version="0.1.0")

class PredictRequest(BaseModel):
    # MVP: 숫자 feature 벡터만 받는 형태(나중에 Avazu feature dict로 확장)
    features: List[float]
    request_id: Optional[str] = None

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": True}

@app.get("/model-info")
def model_info():
    return {
        "model_path": os.getenv("MODEL_PATH", "demo"),
        "model_type": "stub",
        "note": "MVP serving skeleton. Replace stub with real model loader."
    }

@app.post("/predict")
def predict(req: PredictRequest):
    # MVP: 아직 모델 없으니 더미 점수(나중에 로더 연결)
    score = float(sum(req.features)) if req.features else 0.0
    return {"ok": True, "request_id": req.request_id, "score": score}
