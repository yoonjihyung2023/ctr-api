
ctr-api

FastAPI + Docker serving demo for CTR/ML inference.

Why this repo matters

This repository is a lightweight serving proof that complements a model-training project.

It demonstrates:

API design for inference
simple model-serving structure
containerized execution with Docker
recruiter-visible endpoints and response examples
Endpoints
GET /health
GET /model-info
POST /predict
Quickstart
Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8000
Run with Docker
docker build -t ctr-api .
docker run -p 8000:8000 ctr-api
API examples
1) Health check
curl http://127.0.0.1:8000/health

Example response:

{
  "status": "ok"
}
2) Model info
curl http://127.0.0.1:8000/model-info

Example response:

{
  "model_name": "demo_ctr_model",
  "device": "cpu",
  "model_path": "demo"
}
3) Prediction
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"features\":[1,2,3],\"request_id\":\"demo\"}"

Example response:

{
  "request_id": "demo",
  "score": 6.0
}
PowerShell examples
Health
irm http://127.0.0.1:8000/health | ConvertTo-Json
Model info
irm http://127.0.0.1:8000/model-info | ConvertTo-Json
Predict
$body = @{ features = @(1,2,3); request_id = "demo" } | ConvertTo-Json
irm "http://127.0.0.1:8000/predict" -Method POST -ContentType "application/json" -Body $body | ConvertTo-Json
Typical structure
app/
  main.py
Dockerfile
README.md
What this proves

This repo is intentionally small, but it shows the core serving signal:

expose inference endpoints
return structured JSON
package the service with Docker
provide easy local verification
Best paired with

This project is strongest when paired with ctr-seqrec-avazu, where the training/evaluation side is shown.

Together, the two repositories communicate:

train/evaluate a model
serve a model
show reproducible evidence
