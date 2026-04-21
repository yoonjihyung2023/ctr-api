![CI](https://github.com/yoonjihyung2023/ctr-api/actions/workflows/ci.yml/badge.svg?branch=main)


ctr-api

FastAPI + Docker serving demo for CTR-style model inference.

This repository is a lightweight serving proof that complements ctr-seqrec-avazu.

Why this repo matters
Shows a simple model serving API
Uses FastAPI for inference endpoints
Includes Docker workflow for reproducible local serving
Gives recruiter-visible proof beyond offline training metrics
API endpoints
GET /health
GET /model-info
POST /predict
Quickstart
Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8000
Run with Docker
docker build -t ctr-api .
docker run -p 8000:8000 ctr-api
Example requests
Health check
curl http://127.0.0.1:8000/health

Expected response:

{
  "status": "ok"
}
Model info
curl http://127.0.0.1:8000/model-info

Expected response:

{
  "model_path": "demo",
  "device": "cpu"
}
Predict
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"features\":[1,2,3],\"request_id\":\"demo\"}"

Expected response:

{
  "request_id": "demo",
  "score": 6.0
}
PowerShell test examples
irm http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5
irm http://127.0.0.1:8000/model-info | ConvertTo-Json -Depth 5
$body = @{ features = @(1,2,3); request_id = "demo" } | ConvertTo-Json
irm http://127.0.0.1:8000/predict -Method POST -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
Repository structure
app/
  main.py
Dockerfile
README.md
Positioning

This repo is meant to show a practical serving layer for an ML portfolio:

offline model evidence: ctr-seqrec-avazu
online inference demo: ctr-api
Related project
ctr-seqrec-avazu
: leakage-safe CTR benchmark with reproducible metrics
