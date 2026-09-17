# Nigeria AQI Forecast — Streamlit App

Interactive deployment of the *Nigeria Air Quality Forecasting* capstone project.
Pick **National**, a **State**, or a **City**, and the app trains/evaluates 6
forecasting models on the fly, picks the best one, and shows a 12-month-ahead
AQI forecast with confidence intervals.

## Project structure

```
nigeria_aqi_app/
├── app.py              # Streamlit UI
├── forecasting.py       # Model training / evaluation / forecasting logic (no UI code)
├── data/
│   └── nigeria_air_quality_2014_2025.csv
├── requirements.txt
└── README.md
```

`forecasting.py` has zero Streamlit dependencies on purpose — it's the same
methodology as the capstone notebook, refactored into reusable functions, so
you can also import it into a notebook, a batch job, or a future API without
any changes.

---

## 1. Run it locally (2 minutes)

```bash
cd nigeria_aqi_app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will open `http://localhost:8501` in your browser automatically.

---

## 2. Deploy for free — Streamlit Community Cloud (recommended for a hackathon demo)

This is the fastest path to a public URL you can put on your submission / slides.

1. **Push this folder to a public GitHub repo** (Streamlit Cloud deploys from GitHub):
   ```bash
   cd nigeria_aqi_app
   git init
   git add .
   git commit -m "Nigeria AQI forecast app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/nigeria-aqi-forecast.git
   git push -u origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **"New app"** → select your repo → branch `main` → main file path `app.py`.
4. Click **Deploy**. Build takes ~2–5 minutes (installing statsmodels/xgboost/pmdarima).
5. You'll get a public URL like `https://nigeria-aqi-forecast.streamlit.app` — share this
   directly with hackathon judges.

**Tip:** keep the bundled CSV in `data/` inside the repo (it's ~1.4 MB, well under
GitHub's limits) so the deployed app has data out of the box. The in-app uploader
still lets anyone refresh it with newer data without redeploying.

---

## 3. Alternative hosting options

| Platform | Best for | Notes |
|---|---|---|
| **Streamlit Community Cloud** | Fastest hackathon demo | Free, zero server config, GitHub-connected |
| **Hugging Face Spaces** (Streamlit SDK) | Alternative free option, good visibility | Push to a HF repo instead of GitHub |
| **Render / Railway** | If you outgrow Streamlit Cloud limits | Free tier, needs a `Dockerfile` (below) |
| **Docker + any cloud VM** | Full control, production-grade | See Dockerfile below |

### Minimal Dockerfile (for Render/Railway/any container host)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 4. Keeping the forecast current (retraining)

The app retrains on every selection change already (it's fast — 2–6 seconds per
geography since the dataset is small), so there's no separate "retrain job" needed
for the current dataset size. To bring in new monthly data:

- **Quick/manual:** use the sidebar **"upload an updated CSV"** control — no redeploy needed.
- **Automated:** replace `data/nigeria_air_quality_2014_2025.csv` in the repo (e.g. via a
  scheduled GitHub Action that appends new monitoring data and commits), and Streamlit
  Cloud will pick up the change and redeploy automatically on the next push.

If the dataset grows much larger (e.g., daily data across many more cities), consider
moving from "train on every request" to a **nightly batch job** that pre-trains and
pickles each model with `joblib.dump()`, and have the app load the pickled model
instead of retraining live.

---

## 5. Next steps toward a fuller MLOps setup (optional, for extra hackathon polish)

- Wrap `forecasting.py` in a small **FastAPI** service (`/forecast?level=City&value=Lagos`)
  so other apps/teams can consume predictions as JSON, not just via this dashboard.
- Add a **monitoring page**: log each forecast made, and once real AQI for that month is
  published, compute rolling accuracy to show forecast drift over time.
- Add **CI** (GitHub Actions) to run a smoke test (`streamlit.testing.v1.AppTest`) on every
  push, so a broken change never reaches the deployed app.
