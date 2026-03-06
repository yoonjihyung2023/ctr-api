# ctr-api

**FastAPI + Docker serving demo for CTR-style model inference**  
**Companion repo to [`ctr-seqrec-avazu`](https://github.com/yoonjihyung2023/ctr-seqrec-avazu)**  
Exposes simple inference endpoints: `/health`, `/model-info`, `/predict`

## One-line
A lightweight ML serving demo that packages a trained CTR-style model behind a FastAPI API with Docker.

## Why this repo matters
- **Serving-ready demo** for ML inference
- **FastAPI endpoints** for health check, model info, and prediction
- **Dockerized workflow** for reproducible local deployment
- Complements the offline benchmark in **`ctr-seqrec-avazu`**

## Endpoints
- `GET /health`
- `GET /model-info`
- `POST /predict`

## Quickstart
```bash
docker build -t ctr-api .
docker run -p 8000:8000 ctr-api
curl http://127.0.0.1:8000/health

## PowerShell examples
irm http://127.0.0.1:8000/health
irm http://127.0.0.1:8000/model-info

$body = '{"features":[1,2,3],"request_id":"demo"}'
irm http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $body
Example responses
{"ok": true, "model_loaded": true}
{"ok": true, "request_id": "demo", "score": 6.0}
What this repo shows

Model loading and health check

Simple inference API design

Docker-based reproducible serving demo

Portfolio evidence for ML deployment readiness
