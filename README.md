# ctr-api

**FastAPI + Docker serving demo for CTR-style model inference**  
**Companion repo to [`ctr-seqrec-avazu`](https://github.com/yoonjihyung2023/ctr-seqrec-avazu)**  
**Production-minded:** schema validation + error handling + container run + sample response

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
1. `docker build -t ctr-api .`
2. `docker run -p 8000:8000 ctr-api`
3. `curl http://127.0.0.1:8000/health`

## PowerShell examples
- `irm http://127.0.0.1:8000/health`
- `irm http://127.0.0.1:8000/model-info`

Example POST body:
`@{ features = @(0.1, 0.2, 0.3) } | ConvertTo-Json -Depth 3`

## Sample response
`{ "prediction": 0.7312 }`

## Production-minded API design

This repo is intentionally small, but it demonstrates a few production-minded serving ideas:

- **Schema validation** with FastAPI/Pydantic-style request parsing
- **Clear error handling** for invalid payloads and malformed requests
- **Containerized run path** with Docker for reproducible local deployment
- **Predictable response contract** for downstream clients
- **Separation of concerns** between app entrypoint, API schema, and model-loading logic

## Example request contract
`{ "features": [0.1, 0.2, 0.3] }`

## Example success response
`{ "prediction": 0.7312 }`

## Example error response
`{ "detail": "Invalid request payload" }`

## Local container run
1. `docker build -t ctr-api .`
2. `docker run -p 8000:8000 ctr-api`

## Project goal
This repo is built to show a simple but realistic path from trained model artifact to containerized inference API.
