# ctr-api

**FastAPI + Docker serving demo for CTR-style inference**  
**Companion repo to [ctr-seqrec-avazu](https://github.com/yoonjihyung2023/ctr-seqrec-avazu)**  
Exposes simple inference endpoints: /health, /model-info, /predict

## One-line
A lightweight ML serving demo that packages a CTR-style model behind a FastAPI API with Docker.

## Why this repo matters
- **Serving-ready demo** for ML inference
- **FastAPI endpoints** for health, model info, and prediction
- **Dockerized workflow** for reproducible local deployment
- Complements the offline benchmark in **ctr-seqrec-avazu**

## Endpoints
- GET /health
- GET /model-info
- POST /predict

## Quickstart
`ash
docker build -t ctr-api .
docker run -p 8000:8000 ctr-api
curl http://127.0.0.1:8000/health
PowerShell examples
irm http://127.0.0.1:8000/health
irm http://127.0.0.1:8000/model-info

