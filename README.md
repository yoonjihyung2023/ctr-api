# ctr-api
FastAPI + Docker serving demo for CTR/RecSys models

## Run (Docker)
```bash
docker build -t ctr-api .
docker run --rm -p 8000:8000 ctr-api
Run (Local)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
Quick checks (Windows PowerShell)
(irm http://127.0.0.1:8000/health) | ConvertTo-Json -Depth 10
(irm http://127.0.0.1:8000/model-info) | ConvertTo-Json -Depth 10

$body = @{ features = @(1,2,3); request_id = "demo" } | ConvertTo-Json
(irm http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $body) | ConvertTo-Json -Depth 10
Endpoints

GET /health

GET /model-info

POST /predict (body: { "features": [..], "request_id": "..." })

🔮 /predict (request/response)
Request (PowerShell)
$body = '{"features":[1,2,3],"request_id":"demo"}'
irm http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $body
Response (example)
{
  "ok": true,
  "request_id": "demo",
  "score": 6.0
}

