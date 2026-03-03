from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
import torch

from api.model import build_model_from_state_dict

app = FastAPI(title="CTR API (NeuMF from checkpoint)")

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pth")

load_info = {"loaded": False, "error": None, "shapes": None}
model = None

def load_model():
    global model, load_info
    try:
        state = torch.load(MODEL_PATH, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError(f"Expected state_dict (dict), got {type(state)}")

        shapes = {}
        for k in [
            "user_embedding_mf.weight",
            "item_embedding_mf.weight",
            "user_embedding_mlp.weight",
            "item_embedding_mlp.weight",
            "genre_embeddig.weight",
            "mlp_layers.0.weight",
            "mlp_layers.3.weight",
            "affine_output.weight",
        ]:
            if k in state:
                shapes[k] = list(state[k].shape)
        load_info["shapes"] = shapes

        model = build_model_from_state_dict(state)
        model.load_state_dict(state, strict=True)
        model.eval()

        load_info["loaded"] = True
        load_info["error"] = None
    except Exception as e:
        load_info["loaded"] = False
        load_info["error"] = str(e)

load_model()

class PredictRequest(BaseModel):
    user_id: int = Field(..., ge=0)
    item_id: int = Field(..., ge=0)
    genre_id: int = Field(0, ge=0)

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": load_info["loaded"]}

@app.get("/model-info")
def model_info():
    return {
        "model_path": MODEL_PATH,
        "loaded": load_info["loaded"],
        "error": load_info["error"],
        "shapes": load_info["shapes"],
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None or not load_info["loaded"]:
        return {"error": "model not loaded", "detail": load_info["error"]}

    num_users = model.user_embedding_mf.num_embeddings
    num_items = model.item_embedding_mf.num_embeddings
    num_genres = model.genre_embeddig.num_embeddings

    if req.user_id >= num_users:
        return {"error": "user_id out of range", "max_user_id": num_users - 1}
    if req.item_id >= num_items:
        return {"error": "item_id out of range", "max_item_id": num_items - 1}
    if req.genre_id >= num_genres:
        return {"error": "genre_id out of range", "max_genre_id": num_genres - 1}

    p = model.predict_proba(req.user_id, req.item_id, req.genre_id)
    return {"p_click": round(p, 8)}
