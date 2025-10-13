# Stock Dashboard (Streamlit)

A Streamlit dashboard for macro and equity signals using FRED and Yahoo Finance data.

## Quick Start (Local)
- Python 3.10+ recommended.
- Install deps: `pip install -r requirements.txt`
- Add your FRED API key:
  - Option A (env): create a `.env` next to `app.py` with `FRED_API_KEY=your32charlowercasekey`
  - Option B (Streamlit secrets): copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in the key
- Run: `streamlit run app.py`

## Deploy to Streamlit Cloud
1) Push this repo to GitHub (done).
2) Open https://streamlit.io/cloud and sign in with GitHub.
3) New app → select repo `PattuA/stock_dashboard`, branch `main`, app file `app.py`.
4) Click Deploy. The first build may take a few minutes.
5) Add your secrets: App → Settings → Secrets and paste:

```
FRED_API_KEY = "your32charlowercasekey"
```

Notes
- Every push to `main` triggers an automatic redeploy.
- If you need to pin Python, add `runtime.txt` with e.g. `python-3.11`.
- CI runs `pytest` via GitHub Actions on push/PR.

## Project Structure
- `app.py` — Streamlit entrypoint
- `app_context.py` — environment/secrets handling and initial data checks
- `loaders/` — data loader utilities (FRED, Yahoo Finance)
- `panels/` — UI panels (heatmap, timeseries, options, CSP)
- `pages/` — multi-page views

## Secrets
- Required: `FRED_API_KEY` (32 lowercase hex chars). Obtain for free:
  https://research.stlouisfed.org/docs/api/api_key.html
- Locally, set via `.env` or `.streamlit/secrets.toml`.
- In Streamlit Cloud, set under App → Settings → Secrets.

