# ctr-api

**FastAPI + Docker serving demo for CTR-style model inference**  
**Companion repo to [`ctr-seqrec-avazu`](https://github.com/yoonjihyung2023/ctr-seqrec-avazu)**  
Exposes simple inference endpoints: **`/health`**, **`/model-info`**, **`/predict`**

## One-line
A lightweight ML serving demo that packages a trained CTR-style model behind a FastAPI API with Docker.

## Why this repo matters
- **Serving-ready demo** for model inference
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
```

## PowerShell example
```powershell
irm http://127.0.0.1:8000/health
```

## Request example
```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"features\":[0.1,0.2,0.3]}"
```

## Response example
```json
{
  "prediction": 0.7312
}
```

## Project goal
This repo is built to show that a trained CTR-style model can be wrapped as a simple API and run locally in a reproducible Docker-based workflow.
