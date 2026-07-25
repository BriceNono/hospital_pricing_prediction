"""
app.py
------
Flask backend for the Hospital Treatment Pricing Prediction system
(King Faisal Hospital case study, MedPrice AI).

Serves:
  GET  /                -> the prediction UI (templates/index.html)
  POST /predict         -> JSON API that returns a predicted treatment cost
  GET  /health          -> simple health check (useful for Render)

The model is a scikit-learn Pipeline (preprocessing + RandomForestRegressor)
saved with joblib, trained by train_model.py from the same feature set used
in Hospital_Treatment_Pricing_RandomForest.ipynb.
"""

import os
import time
import traceback

import joblib
import pandas as pd
import requests
from flask import Flask, request, jsonify, render_template

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "hospital_treatment_pricing_rf_model.joblib")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# USD -> RWF currency conversion
# ---------------------------------------------------------------------------
# Live rate is fetched from a free, keyless exchange-rate API and cached in
# memory for CACHE_TTL_SECONDS so we don't hit the network on every request.
# If the API is ever unreachable (offline demo, blocked network, etc.) we
# fall back to FALLBACK_USD_TO_RWF so the app keeps working end-to-end.
EXCHANGE_RATE_API_URL = "https://open.er-api.com/v6/latest/USD"
FALLBACK_USD_TO_RWF = 1450.0  # approximate mid-2026 rate, used only if the live fetch fails
CACHE_TTL_SECONDS = 6 * 60 * 60  # refresh at most every 6 hours

_rate_cache = {"rate": None, "fetched_at": 0, "source": "fallback"}


def get_usd_to_rwf_rate():
    """Return (rate, source) where source is 'live' or 'fallback'."""
    now = time.time()
    if _rate_cache["rate"] is not None and (now - _rate_cache["fetched_at"]) < CACHE_TTL_SECONDS:
        return _rate_cache["rate"], _rate_cache["source"]

    try:
        resp = requests.get(EXCHANGE_RATE_API_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        rate = float(data["rates"]["RWF"])
        _rate_cache.update({"rate": rate, "fetched_at": now, "source": "live"})
        return rate, "live"
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not fetch live USD->RWF rate, using fallback: {exc}")
        _rate_cache.update({"rate": FALLBACK_USD_TO_RWF, "fetched_at": now, "source": "fallback"})
        return FALLBACK_USD_TO_RWF, "fallback"

# ---------------------------------------------------------------------------
# Load the trained model once at startup
# ---------------------------------------------------------------------------
model = None
model_load_error = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:  # noqa: BLE001
    model_load_error = str(exc)
    print(f"WARNING: could not load model from {MODEL_PATH}: {exc}")

# Allowed categorical values (mirrors the training dataset / dropdown options)
DIAGNOSES = [
    "Malaria", "Heart Disease", "Kidney Infection", "Typhoid", "COVID-19",
    "Diabetes", "Appendicitis", "Hypertension", "Fracture", "Pneumonia",
]
TREATMENT_TYPES = ["Surgery", "Emergency", "Outpatient", "Inpatient"]
RESOURCE_LEVELS = ["Low", "Medium", "High"]
GENDERS = ["Male", "Female"]
YES_NO = ["Yes", "No"]

REQUIRED_FIELDS = [
    "Age", "Gender", "Diagnosis", "Treatment_Type",
    "Length_of_Stay", "BMI", "Smoker", "Resource_Utilization",
]


def validate_and_build_input(payload: dict):
    """Validate the incoming JSON payload and turn it into a one-row DataFrame.

    Returns (dataframe, None) on success or (None, error_message) on failure.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in payload or payload[f] in (None, "")]
    if missing:
        return None, f"Missing required field(s): {', '.join(missing)}"

    try:
        age = int(payload["Age"])
        length_of_stay = int(payload["Length_of_Stay"])
        bmi = float(payload["BMI"])
    except (TypeError, ValueError):
        return None, "Age and Length_of_Stay must be integers, and BMI must be a number."

    if not (0 <= age <= 120):
        return None, "Age must be between 0 and 120."
    if not (1 <= length_of_stay <= 365):
        return None, "Length_of_Stay must be between 1 and 365 days."
    if not (10 <= bmi <= 70):
        return None, "BMI must be between 10 and 70."

    gender = payload["Gender"]
    diagnosis = payload["Diagnosis"]
    treatment_type = payload["Treatment_Type"]
    smoker = payload["Smoker"]
    resource_utilization = payload["Resource_Utilization"]

    if gender not in GENDERS:
        return None, f"Gender must be one of {GENDERS}."
    if diagnosis not in DIAGNOSES:
        return None, f"Diagnosis must be one of {DIAGNOSES}."
    if treatment_type not in TREATMENT_TYPES:
        return None, f"Treatment_Type must be one of {TREATMENT_TYPES}."
    if smoker not in YES_NO:
        return None, f"Smoker must be one of {YES_NO}."
    if resource_utilization not in RESOURCE_LEVELS:
        return None, f"Resource_Utilization must be one of {RESOURCE_LEVELS}."

    row = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Diagnosis": diagnosis,
        "Treatment_Type": treatment_type,
        "Length_of_Stay": length_of_stay,
        "BMI": bmi,
        "Smoker": smoker,
        "Resource_Utilization": resource_utilization,
    }])
    return row, None


@app.route("/")
def index():
    return render_template(
        "index.html",
        diagnoses=DIAGNOSES,
        treatment_types=TREATMENT_TYPES,
        resource_levels=RESOURCE_LEVELS,
        genders=GENDERS,
    )


@app.route("/health")
def health():
    status = "ok" if model is not None else "model_not_loaded"
    return jsonify({"status": status, "error": model_load_error}), (200 if model is not None else 503)


@app.route("/exchange-rate")
def exchange_rate():
    """Standalone endpoint the frontend can poll to show the rate on its own,
    independent of a prediction (e.g. in a small 'rate used' footnote)."""
    rate, source = get_usd_to_rwf_rate()
    return jsonify({"usd_to_rwf": rate, "source": source})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not available on the server. Please contact the administrator."}), 503

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    row, error = validate_and_build_input(payload)
    if error:
        return jsonify({"error": error}), 400

    try:
        prediction = model.predict(row)[0]
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    cost_usd = round(float(prediction), 2)
    rate, rate_source = get_usd_to_rwf_rate()
    cost_rwf = round(cost_usd * rate, 2)

    return jsonify({
        "predicted_cost_usd": cost_usd,
        "predicted_cost_rwf": cost_rwf,
        "exchange_rate": {"usd_to_rwf": rate, "source": rate_source},
        "input": row.iloc[0].to_dict(),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)