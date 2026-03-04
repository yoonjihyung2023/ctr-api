## ctr-api
FastAPI + Docker serving demo for CTR (click-through rate) models.

## What this repo shows
- Minimal inference API: `/health`, `/model-info`, `/predict`
- Docker build/run (copy-paste friendly)
- Example requests in Windows PowerShell

## Quickstart (Docker)
```bash
docker build -t ctr-api .
docker run --rm -p 8000:8000 ctr-api
```

## Test endpoints (PowerShell)
irm http://127.0.0.1:8000/health
irm http://127.0.0.1:8000/model-info

$body = '{"features":[1,2,3],"request_id":"demo"}'
irm http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $body

## Response examples (what you should see)

GET /health

{"ok":true,"model_loaded":true}

GET /model-info

{"ok":true,"model_path":"models/model.pth","device":"cpu"}

POST /predict

{"ok":true,"request_id":"demo","score":6.0}

Swagger UI: http://127.0.0.1:8000/docs

## Utilities

Inspect a model checkpoint:

py inspect_ckpt.py --path models/model.pth
Related

## Modeling benchmark (leakage-safe CTR/RecSys): https://github.com/yoonjihyung2023/ctr-seqrec-avazu
