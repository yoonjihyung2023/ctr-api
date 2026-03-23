# ctr-api

FastAPI + Docker serving demo for CTR prediction, built as a companion deployment project to `ctr-seqrec-avazu`.

Includes 3 simple endpoints for health check, model metadata, and prediction to demonstrate production-style model serving.

**Endpoints:** `/health`, `/model-info`, `/predict`  
**Run:** Docker build + container start  
**Purpose:** lightweight serving proof for Ads/RecSys ML portfolio

## Quickstart
```bash
docker build -t ctr-api .
docker run -p 8000:8000 ctr-api
curl http://localhost:8000/health
```

## Example
**POST** `/predict` with feature payload -> returns prediction response.
