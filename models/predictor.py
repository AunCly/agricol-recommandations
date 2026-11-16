import json
import os
from pathlib import Path

import joblib
import pandas as pd

from api.agricol_context import AgricolContextModel, RecommendContextModel

base_dir = Path(__file__).resolve().parent.parent

# Conversion : 10 000 hectogrammes par hectare = 1 tonne par hectare
hg_per_tonne = 10_000

# Colonnes attendues par le modèle, dans l'ordre utilisé à l'entraînement
features = [
    "Area",
    "Item",
    "avg_temp",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "pesticides_kg_ha",
    "missing_pesticide_value",
    "missing_rain_value",
    "missing_temp_value",
]

# Colonnes numériques (peuvent être absentes, le pipeline les impute)
numeric_features = [
    "avg_temp",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "pesticides_kg_ha",
]


def build_row(context: RecommendContextModel, item: str) -> dict:
    """Construit une ligne d'entrée du modèle pour une culture donnée."""
    row = {
        "Area": context.area,
        "Item": item,
        "avg_temp": context.avg_temp,
        "average_rain_fall_mm_per_year": context.average_rain_fall_mm_per_year,
        "pesticides_tonnes": context.pesticides_tonnes,
        "pesticides_kg_ha": context.pesticides_kg_ha,
    }

    # Indicateurs de valeurs manquantes (1 = absente, 0 = renseignée)
    if context.pesticides_tonnes is None:
        row["missing_pesticide_value"] = 1
    else:
        row["missing_pesticide_value"] = 0

    if context.average_rain_fall_mm_per_year is None:
        row["missing_rain_value"] = 1
    else:
        row["missing_rain_value"] = 0

    if context.avg_temp is None:
        row["missing_temp_value"] = 1
    else:
        row["missing_temp_value"] = 0

    return row


def format_data_for_ml_model(context: RecommendContextModel, items: list[str]) -> pd.DataFrame:
    """Construit le DataFrame d'entrée du modèle : une ligne par culture."""
    rows = []
    for item in items:
        row = build_row(context, item)
        rows.append(row)

    data = pd.DataFrame(rows)

    # Les valeurs absentes (None) doivent devenir des NaN de type float
    for column in numeric_features:
        data[column] = data[column].astype("float64")

    # On remet les colonnes dans l'ordre attendu par le modèle
    data = data[features]

    return data


def load_artifact(path: Path):
    """Charge un artefact joblib en vérifiant qu'il existe."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artefact introuvable : {path}")
    return joblib.load(path)


def load_metadata(path: Path) -> dict:
    """Charge le fichier de métadonnées JSON du modèle."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Métadonnées introuvables : {path}")
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def format_yield(yield_hg_ha: float) -> dict:
    """Retourne le rendement en hg/ha et en t/ha."""
    yield_t_ha = yield_hg_ha / hg_per_tonne
    return {
        "yield_hg_ha": yield_hg_ha,
        "yield_t_ha": yield_t_ha,
    }


class AgricultureYieldPredictor:
    def __init__(self, model_name: str = "agriculture_model"):
        model_path = base_dir / "models" / f"{model_name}.joblib"
        metadata_path = base_dir / "models" / f"{model_name}_metadata.json"

        self.model = load_artifact(model_path)
        self.metadata = load_metadata(metadata_path)

        # Liste des cultures connues par le modèle
        self.cultures = self.metadata["cultures"]

    def predict(self, context: AgricolContextModel) -> dict:
        """Prédit le rendement pour la culture demandée dans le contexte."""
        data = format_data_for_ml_model(context, [context.item])

        predictions = self.model.predict(data)
        yield_hg_ha = float(predictions[0])

        return format_yield(yield_hg_ha)

    def recommend(self, context: RecommendContextModel) -> list[dict]:
        """Prédit le rendement de chaque culture et les classe par rendement décroissant."""
        # Une seule prédiction pour toutes les cultures (une ligne par culture)
        data = format_data_for_ml_model(context, self.cultures)
        predictions = self.model.predict(data)

        # Association de chaque culture à son rendement prédit
        results = []
        for index, item in enumerate(self.cultures):
            yield_hg_ha = float(predictions[index])

            result = {"item": item}
            result.update(format_yield(yield_hg_ha))
            results.append(result)

        # Tri du meilleur au moins bon rendement
        results = sorted(results, key=lambda result: result["yield_hg_ha"], reverse=True)

        # Ajout du rang (1 = meilleur rendement)
        ranking = []
        for rank, result in enumerate(results, start=1):
            ranked_result = {"rank": rank}
            ranked_result.update(result)
            ranking.append(ranked_result)

        return ranking
