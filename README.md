# MedPrice AI — Hospital Treatment Pricing Prediction

A deployable Flask web app for the *Hospital Treatment Pricing Predictions*
project (King Faisal Hospital case study). It wraps the Random Forest
pipeline from `Hospital_Treatment_Pricing_RandomForest.ipynb` behind a
simple, clean web interface where an administrator enters patient/treatment
details and receives a predicted treatment cost in USD.

## Project structure

```
hospital-pricing-app/
├── app.py                  # Flask app: serves the UI and the /predict API
├── train_model.py          # Trains the RF pipeline and saves the .joblib model
├── requirements.txt        # Python dependencies
├── Procfile                # For Render/Heroku-style process declaration
├── render.yaml             # One-click Render.com blueprint
├── data/
│   └── hospital_treatment_pricing_dataset.csv
├── model/
│   └── hospital_treatment_pricing_rf_model.joblib   # created by train_model.py
├── templates/
│   └── index.html          # Main page (Jinja2 template)
└── static/
    ├── style.css
    └── script.js
```

## How the model works

Mirrors the notebook exactly:

- **Numeric features** (`Age`, `Length_of_Stay`, `BMI`) — passed through unchanged.
- **Ordinal feature** (`Resource_Utilization`: Low < Medium < High) — `OrdinalEncoder`.
- **Nominal features** (`Gender`, `Diagnosis`, `Treatment_Type`, `Smoker`) — `OneHotEncoder`.
- **Model**: `RandomForestRegressor` inside a scikit-learn `Pipeline`, so
  preprocessing + inference are one saved object (`joblib`).

On the sample dataset provided, this configuration scores roughly
**MAE ≈ 193 USD, R² ≈ 0.91** on a held-out test split (your numbers may vary
slightly depending on random splits/params). If you want the full
`GridSearchCV` sweep from the notebook instead of the fast fixed-parameter
version, set `FULL_GRID_SEARCH = True` at the top of `train_model.py` (this
takes noticeably longer to run).

## Run locally in VS Code

1. Open the `hospital-pricing-app` folder in VS Code.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Train the model (creates `model/hospital_treatment_pricing_rf_model.joblib`):
   ```bash
   python train_model.py
   ```
5. Run the app:
   ```bash
   python app.py
   ```
6. Open **http://localhost:5000** in your browser.

## Deploying to Render

1. Push this folder to a GitHub repository.
2. In Render, choose **New → Blueprint** and point it at the repo — Render
   will read `render.yaml` automatically and configure:
   - Build command: `pip install -r requirements.txt && python train_model.py`
   - Start command: `gunicorn app:app`
3. Alternatively, create a **New → Web Service** manually with the same
   build/start commands above and no extra environment variables required.
4. Once deployed, Render gives you a public URL serving the same UI.

> The build step re-runs `train_model.py` on every deploy so the model file
> is always freshly built from `data/hospital_treatment_pricing_dataset.csv`
> — you don't need to commit the `.joblib` file, though it's fine to if you
> prefer a faster build (just make sure `model/` exists in the repo).

## API reference

### `GET /health`
Returns `{"status": "ok"}` if the model loaded successfully.

### `POST /predict`
Body (JSON):
```json
{
  "Age": 45,
  "Gender": "Male",
  "Diagnosis": "Diabetes",
  "Treatment_Type": "Inpatient",
  "Length_of_Stay": 10,
  "BMI": 27.5,
  "Smoker": "No",
  "Resource_Utilization": "Medium"
}
```
Response:
```json
{
  "predicted_cost_usd": 1553.34,
  "input": { "...": "echoed, validated input" }
}
```

## Notes on scope vs. the full dissertation design

Chapter Four of the dissertation describes a larger target architecture
(React frontend, FastAPI backend, JWT auth, MySQL/PostgreSQL, role-based
access). This deliverable implements the **core predictive feature** (FR2–FR4
from the functional requirements table) as a lightweight, fully deployable
Flask + vanilla HTML/CSS/JS app — the fastest path to a working, hosted demo.
Authentication, historical record storage/export (FR5–FR8), and a relational
database can be layered on top of this same `app.py` without changing the
prediction endpoint.
# maize_leaves_disease
# hospital_pricing_prediction
# hospital_pricing_prediction
