# Live API Verification

Checked at: 2026-05-28 19:50:07 +09:00

## Endpoints

- Health: https://ctr-api.onrender.com/health
- Swagger UI: https://ctr-api.onrender.com/docs
- Model info: https://ctr-api.onrender.com/model-info

## Result

| Endpoint | OK |
|---|---:|
| /health | True |
| /docs | True |
| /model-info | True |

## /health response

{"ok":true,"model_loaded":true}

## /model-info response

{"model_path":"demo","model_type":"stub","note":"MVP serving skeleton. Replace stub with real model loader."}
