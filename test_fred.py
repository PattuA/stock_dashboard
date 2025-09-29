import os, json, urllib.request
from pathlib import Path
from dotenv import load_dotenv

# Load .env that sits next to this file (works regardless of working dir)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

k = os.getenv("FRED_API_KEY", "")
print("Key OK:", len(k) == 32 and k.isalnum() and k.islower(), "mask:", (k[:4]+"..."+k[-4:]) if k else "(empty)")

url = f"https://api.stlouisfed.org/fred/series/observations?series_id=M2SL&api_key={k}&file_type=json&limit=1"
with urllib.request.urlopen(url, timeout=10) as r:
    data = json.loads(r.read().decode())
    print("HTTP OK. Latest record keys:", list(data.keys())[:5])
