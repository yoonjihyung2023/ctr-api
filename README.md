# ctr-api

[![CI](https://github.com/yoonjihyung2023/ctr-api/actions/workflows/ci.yml/badge.svg)](https://github.com/yoonjihyung2023/ctr-api/actions/workflows/ci.yml)

FastAPI + Docker inference API for CTR-style prediction.

Live Demo: https://ctr-api.onrender.com/docs

## What this project proves

This project shows that I can serve an ML-style prediction model as a real API, not just train offline notebooks.

It demonstrates:

- FastAPI inference API
- Dockerized serving
- Health check endpoint
- Model info endpoint
- Prediction endpoint
- GitHub Actions CI
- Live Swagger UI demo

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Check API and model status |
| GET | /model-info | Show model path and device |
| POST | /predict | Return CTR-style prediction score |

## Quickstart

Build and run:

    docker build -t ctr-api .
    docker run -p 8000:8000 ctr-api

Open Swagger UI:

    http://127.0.0.1:8000/docs

## Example

Health check:

    curl http://127.0.0.1:8000/health

Expected response:

    {"ok":true,"model_loaded":true}

Model info:

    curl http://127.0.0.1:8000/model-info

Prediction endpoint:

    POST /predict

Live Swagger UI:

    https://ctr-api.onrender.com/docs

## Portfolio signal

For recruiters, this repo proves basic production-oriented ML serving ability:

- offline model concept to API endpoint
- API endpoint to Docker container
- Docker container to live demo
- GitHub repo to CI-visible project

Related benchmark repo:

https://github.com/yoonjihyung2023/ctr-seqrec-avazu

## Tech stack

- Python
- FastAPI
- PyTorch
- Docker
- Uvicorn
- GitHub Actions


## Sample prediction request

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"request_id\":\"demo\",\"features\":[1,2,3]}"
```

Expected response:

```json
{"ok":true,"request_id":"demo","score":6.0}
```

