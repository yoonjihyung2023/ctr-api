# ctr-api

FastAPI + Docker serving demo for CTR-style model inference  
Companion repo to [`ctr-seqrec-avazu`](https://github.com/yoonjihyung2023/ctr-seqrec-avazu)  
Endpoints: `/health`, `/model-info`, `/predict`

## One-line
A lightweight ML serving demo that packages a CTR-style model behind a FastAPI API with Docker.

## Why this repo matters
- Shows a serving-ready demo for ML inference
- Uses schema validation and simple error handling
- Supports reproducible local deployment with Docker
- Complements the offline benchmark story in ctr-seqrec-avazu

## API surface
- `GET /health`
- `GET /model-info`
- `POST /predict`

## Quickstart
```bash
docker build -t ctr-api .
docker run -p 8000:8000 ctr-api
curl http://127.0.0.1:8000/health
Windows PowerShell
docker build -t ctr-api .
docker run -p 8000:8000 ctr-api
irm http://127.0.0.1:8000/health
irm http://127.0.0.1:8000/model-info
Example request
{
  "features": {
    "hour": 14,
    "banner_pos": 0,
    "site_id": "example_site",
    "device_type": 1
  }
}
Example response
{
  "click_probability": 0.1234
}
Production-minded points

Request schema validation

Clear endpoint separation

Simple inference contract

Dockerized local serving workflow
