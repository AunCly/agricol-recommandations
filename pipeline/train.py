"""Pipeline de réentraînement du modèle de rendement.

Reprend le modèle retenu au cycle 6 (notebooks/training_cycle6.ipynb) et enchaîne :

    1. chargement des données
    2. entraînement sur les années passées (train)
    3. évaluation sur les années récentes (test), comparée à la baseline naïve
    4. validation : le nouveau modèle n'est accepté que s'il passe les contrôles
    5. réentraînement sur toutes les années et sauvegarde pour l'API

Si la validation échoue, rien n'est écrit : le modèle actuel est conservé.
La décision est transmise au workflow GitHub Actions (sortie « accepted »),
qui ne publie et ne déploie le modèle que s'il a été accepté.

Lancement, depuis la racine du projet :
    uv run python -m pipeline.train
"""

import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

import notebooks.utils as utils
# Même chemin d'import que dans le notebook : le modèle sauvegardé référence cette classe
from notebooks.agritech_feature_engineering import AgritechFeatureEngineering

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "rafined" / "dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "agriculture_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "agriculture_model_metadata.json"

INPUT_FEATURES = [
    "Area", "Item",
    "avg_temp", "average_rain_fall_mm_per_year", "pesticides_tonnes", "pesticides_kg_ha",
    "missing_pesticide_value", "missing_rain_value", "missing_temp_value",
]
TARGET = "hg/ha"

# Hyperparamètres retenus au cycle 6
MODEL_PARAMS = {
    "n_estimators": 100,
    "min_samples_leaf": 2,
    "max_features": 0.5,
    "max_depth": 40,
    "bootstrap": False,
}


# ---------------------------------------------------------------------------
# 1. Données
# ---------------------------------------------------------------------------

def load_data():
    """Charge le dataset et le découpe en train (passé) / test (années récentes)."""
    dataset = pd.read_csv(DATASET_PATH)
    X_train, X_test, y_train, y_test = utils.test_train_time_split(dataset)
    return dataset, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 2. Modèle
# ---------------------------------------------------------------------------

def build_model():
    """Pipeline complet : feature engineering, prétraitement, Random Forest sur cible log1p."""
    numeric_features = INPUT_FEATURES[2:] + ["aridity", "avg_temp_square"]

    preprocessor = ColumnTransformer([
        ("area_encoding", TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=42)), ["Area"]),
        ("item_encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Item"]),
        ("num_pipeline", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
    ])

    pipeline = Pipeline([
        ("features", AgritechFeatureEngineering()),
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(**MODEL_PARAMS, random_state=42, n_jobs=-1)),
    ])

    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def train(X, y):
    model = build_model()
    model.fit(X[INPUT_FEATURES], y)
    return model


# ---------------------------------------------------------------------------
# 3. Évaluation
# ---------------------------------------------------------------------------

def baseline_predictions(X_train, y_train, X_test):
    """Baseline naïve (cycle 0) : rendement médian historique du couple (pays, culture)."""
    train_df = X_train.assign(**{TARGET: y_train})
    median_area_item = train_df.groupby(["Area", "Item"])[TARGET].median().rename("pred")
    median_item = train_df.groupby("Item")[TARGET].median()

    predictions = X_test.join(median_area_item, on=["Area", "Item"])["pred"]
    # Couple jamais vu : médiane de la culture, tous pays confondus
    return predictions.fillna(X_test["Item"].map(median_item))


def evaluate(model, X_train, y_train, X_test, y_test):
    """Scores du modèle et de la baseline sur le même jeu de test."""
    model_scores = utils.evaluate(y_test, model.predict(X_test[INPUT_FEATURES]))
    baseline_scores = utils.evaluate(y_test, baseline_predictions(X_train, y_train, X_test))
    return {
        "modele": {name: round(float(value), 4) for name, value in model_scores.items()},
        "baseline": {name: round(float(value), 4) for name, value in baseline_scores.items()},
    }


# ---------------------------------------------------------------------------
# 4. Validation
# ---------------------------------------------------------------------------

def load_current_rmse():
    """RMSE du modèle actuellement en production (None s'il n'y en a pas)."""
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH, encoding="utf-8") as file:
        return json.load(file)["evaluation"]["modele"]["rmse"]


def validate(evaluation, current_rmse):
    """Liste des contrôles et de leur résultat (True = réussi).

    Le nouveau modèle doit avoir une RMSE strictement plus basse que la baseline
    et que le modèle actuellement en production.
    """
    new_rmse = evaluation["modele"]["rmse"]

    checks = {
        f"RMSE {new_rmse:.0f} < baseline {evaluation['baseline']['rmse']:.0f}": new_rmse < evaluation["baseline"]["rmse"],
    }
    if current_rmse is not None:
        checks[f"RMSE {new_rmse:.0f} < modèle actuel {current_rmse:.0f}"] = new_rmse < current_rmse

    return checks


# ---------------------------------------------------------------------------
# 5. Sauvegarde
# ---------------------------------------------------------------------------

def build_metadata(dataset, evaluation):
    """Métadonnées lues par l'API et le dashboard."""
    return {
        "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "cultures": sorted(dataset["Item"].unique()),
        "pays": sorted(dataset["Area"].unique()),
        # Cultures observées dans chaque pays : la recommandation ne propose que celles-ci
        "cultures_par_pays": {area: sorted(items.unique()) for area, items in dataset.groupby("Area")["Item"]},
        # Bornes des variables numériques, utilisées pour les sliders du dashboard
        "feature_ranges": {
            column: {
                "min": round(float(dataset[column].min()), 2),
                "median": round(float(dataset[column].median()), 2),
                "max": round(float(dataset[column].max()), 2),
            }
            for column in ["avg_temp", "average_rain_fall_mm_per_year", "pesticides_tonnes", "pesticides_kg_ha"]
        },
        "training_years": [int(dataset["Year"].min()), int(dataset["Year"].max())],
        "model_params": MODEL_PARAMS,
        "evaluation": evaluation,
    }


def save(model, metadata):
    joblib.dump(model, MODEL_PATH, compress=3)
    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def log_to_mlflow(evaluation, accepted):
    """Trace l'exécution dans MLflow, comme les cycles des notebooks."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR / 'mlflow.db'}"))
    mlflow.set_experiment("Agriculture_optimization")
    with mlflow.start_run(run_name="Pipeline - réentraînement"):
        mlflow.set_tags({"cycle": "pipeline", "accepte": str(accepted)})
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_metrics({f"test_{name}": value for name, value in evaluation["modele"].items()})
        mlflow.log_metrics({f"baseline_{name}": value for name, value in evaluation["baseline"].items()})


def share_decision(accepted):
    """Transmet la décision aux steps suivants du workflow GitHub Actions."""
    github_output = os.getenv("GITHUB_OUTPUT")  # n'existe que dans GitHub Actions
    if github_output:
        with open(github_output, "a") as file:
            file.write(f"accepted={str(accepted).lower()}\n")


# ---------------------------------------------------------------------------

def main():
    print("1. Chargement des données")
    dataset, X_train, X_test, y_train, y_test = load_data()

    print("2. Entraînement sur le train")
    model = train(X_train, y_train)

    print("3. Évaluation sur le test")
    evaluation = evaluate(model, X_train, y_train, X_test, y_test)
    print(pd.DataFrame(evaluation).T)

    print("4. Validation")
    checks = validate(evaluation, load_current_rmse())
    for name, passed in checks.items():
        print(f"   [{'OK' if passed else 'ÉCHEC'}] {name}")
    accepted = all(checks.values())
    log_to_mlflow(evaluation, accepted)
    share_decision(accepted)

    if not accepted:
        print("Modèle refusé : le modèle actuel est conservé.")
        return

    print("5. Réentraînement sur toutes les années et sauvegarde")
    final_model = train(dataset, dataset[TARGET])
    save(final_model, build_metadata(dataset, evaluation))
    print(f"Modèle enregistré : {MODEL_PATH}")


if __name__ == "__main__":
    main()
