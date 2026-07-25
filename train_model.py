"""
train_model.py
---------------
Trains the Random Forest pipeline for the Hospital Treatment Pricing
Prediction system (King Faisal Hospital case study) and saves it to
model/hospital_treatment_pricing_rf_model.joblib.

This mirrors the pipeline built in Hospital_Treatment_Pricing_RandomForest.ipynb:
  - Numeric features (Age, Length_of_Stay, BMI) -> passthrough
  - Ordinal feature (Resource_Utilization: Low < Medium < High) -> OrdinalEncoder
  - Nominal features (Gender, Diagnosis, Treatment_Type, Smoker) -> OneHotEncoder
  - Model: RandomForestRegressor, tuned with GridSearchCV

Run this once locally (or during the Render build step) to produce the
model file that app.py loads at request time:

    python train_model.py
"""

import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
DATA_PATH = "data/hospital_treatment_pricing_dataset.csv"
MODEL_PATH = "model/hospital_treatment_pricing_rf_model.joblib"

NOMINAL_FEATURES = ["Gender", "Diagnosis", "Treatment_Type", "Smoker"]
ORDINAL_FEATURES = ["Resource_Utilization"]
NUMERIC_FEATURES = ["Age", "Length_of_Stay", "BMI"]
RESOURCE_ORDER = [["Low", "Medium", "High"]]
TARGET = "Treatment_Cost_USD"

# Set to True for the full GridSearchCV sweep from the notebook (slower).
# Set to False for a quick, still-solid default RF (fast, good for redeploys).
FULL_GRID_SEARCH = False


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df_model = df.drop(columns=["Patient_ID"])
    df_model = df_model.drop_duplicates()
    df_model = df_model.dropna(subset=[TARGET])
    return df_model


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("ord", OrdinalEncoder(categories=RESOURCE_ORDER), ORDINAL_FEATURES),
            ("nom", OneHotEncoder(handle_unknown="ignore"), NOMINAL_FEATURES),
        ]
    )
    model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main():
    start = time.time()
    print("Loading and cleaning dataset...")
    df_model = load_and_clean_data(DATA_PATH)

    X = df_model.drop(columns=[TARGET])
    y = df_model[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()

    if FULL_GRID_SEARCH:
        print("Running GridSearchCV (this can take a while)...")
        param_grid = {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        }
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)
        print("Best parameters:", grid_search.best_params_)
        best_model = grid_search.best_estimator_
    else:
        print("Training Random Forest with fixed, notebook-recommended parameters...")
        pipeline.fit(X_train, y_train)
        best_model = pipeline

    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nModel performance on held-out test set:")
    print(f"  MAE : {mae:.2f} USD")
    print(f"  RMSE: {rmse:.2f} USD")
    print(f"  R2  : {r2:.4f}")

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
