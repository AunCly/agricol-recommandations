import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from api.agricol_context import AgricolContextModel, RecommendContextModel
from models.predictor import AgricultureYieldPredictor

load_dotenv()
base_dir = Path(__file__).resolve().parent.parent

api_key_header = APIKeyHeader(name="x-api-key")
predictor = AgricultureYieldPredictor()

app = FastAPI(
    title="Api de prédiction du risque de rendement agricole.",
    description="Cette API permet de prédire le rendement agricole pour un contexte donné.",
    version="0.0.1",
    openapi_tags=[
        {
            "name": "Prédiction de rendement agricole",
            "description": "Endpoints pour prédire le rendement agricole"
        }
    ],
    contact={
        "email": "aurelien.clugery@gmail.com",
    }
)

async def verify_api_key(api_key: str = Security(api_key_header)):

    key = os.getenv('API_KEY')

    if not key or api_key != key:
        raise HTTPException(status_code=403)

    return api_key

@app.get(
    "/health",
    summary="Vérifier la santé de l'API",
    responses={
        200: {"description": "Le serveur de l'API fonctionne correctement"},
    }
)
def health_endpoint():
    return {"message": "Alive !"}

@app.get(
    "/metadata",
    dependencies=[Depends(verify_api_key)],
    summary="Informations nécessaires au front-end (cultures, pays, bornes des variables, performances).",
    responses={
        200: {"description": "Métadonnées récupérées avec succès"},
        403: {"description": "Clé API manquante ou invalide"},
    }
)
def metadata_endpoint():
    metadata = predictor.metadata
    return {
        "version": metadata["version"],
        "cultures": metadata["cultures"],
        "pays": metadata["pays"],
        "feature_ranges": metadata["feature_ranges"],
        "training_years": metadata["training_years"],
        "evaluation": metadata["evaluation"]["modele"],
    }

@app.post(
    "/predict",
    dependencies=[Depends(verify_api_key)],
    summary="Prédiction du rendement pour le contexte donné.",
    responses={
        200: {"description": "Prédiction réussie"},
        401: {"description": "Clé API manquante ou invalide"},
        422: {"description": "Données d'employé invalides (erreur de validation Pydantic)"}
    }
)
def predict_endpoint(application: AgricolContextModel):
    result = predictor.predict(application)
    return result

@app.post(
    "/recommend",
    dependencies=[Depends(verify_api_key)],
    summary="Classement des cultures par rendement prédit décroissant pour le contexte donné.",
    responses={
        200: {"description": "Recommandation récupérée avec succès"},
        401: {"description": "Clé API manquante ou invalide"},
        422: {"description": "Contexte invalide (erreur de validation Pydantic)"}
    }
)
def recommend_endpoint(context: RecommendContextModel):
    return predictor.recommend(context)