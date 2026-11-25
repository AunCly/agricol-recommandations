import os
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from api.main import app

load_dotenv()

client = TestClient(app)
headers = {"x-api-key": os.getenv('API_KEY')}

@pytest.fixture
def get_random_context():
    base_dir = Path(__file__).resolve().parent.parent
    data = pd.read_csv(base_dir / 'data/rafined/dataset.csv')

    row = data.sample(1).iloc[0]

    # Les NaN ne sont pas sérialisables en JSON : on les remplace par None
    def value(column):
        if pd.isna(row[column]):
            return None
        return float(row[column])

    return {
        "area": row["Area"],
        "item": row["Item"],
        "average_rain_fall_mm_per_year": value("average_rain_fall_mm_per_year"),
        "pesticides_tonnes": value("pesticides_tonnes"),
        "pesticides_kg_ha": value("pesticides_kg_ha"),
        "avg_temp": value("avg_temp"),
    }


class TestApi:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"message": "Alive !"}

    def test_predict_with_wrong_api_key(self):
        response = client.post("/predict", headers={"x-api-key": "foo"})
        assert response.status_code == 403
        assert response.json() == {"detail": "Forbidden"}

    def test_metadata(self):
        response = client.get("/metadata", headers=headers)

        response_keys_needed = {"version", "cultures", "pays", "feature_ranges", "training_years", "evaluation"}

        assert response.status_code == 200
        assert response_keys_needed == response.json().keys()

    def test_predict(self, get_random_context):
        response = client.post("/predict", json=get_random_context, headers=headers)

        response_keys_needed = {"yield_hg_ha", "yield_t_ha"}

        assert response.status_code == 200
        assert response_keys_needed == response.json().keys()

    def test_predict_with_missing_fields(self, get_random_context):
        random_context = get_random_context

        del(random_context['item'])

        response = client.post("/predict", json=random_context, headers=headers)

        assert response.status_code == 422

    def test_recommend(self, get_random_context):
        random_context = get_random_context

        del(random_context['item'])

        response = client.post("/recommend", json=random_context, headers=headers)

        response_keys_needed = {"rank", "item", "yield_hg_ha", "yield_t_ha"}

        assert response.status_code == 200
        results = response.json()
        assert isinstance(results, list)
        for result in results:
            assert response_keys_needed == result.keys()

    def test_recommend_only_country_crops(self, get_random_context):
        base_dir = Path(__file__).resolve().parent.parent
        data = pd.read_csv(base_dir / 'data/rafined/dataset.csv')

        random_context = get_random_context
        del(random_context['item'])
        country_crops = set(data.loc[data['Area'] == random_context['area'], 'Item'])

        response = client.post("/recommend", json=random_context, headers=headers)

        assert response.status_code == 200
        assert {result['item'] for result in response.json()} == country_crops

    def test_recommend_with_missing_fields(self, get_random_context):
        random_context = get_random_context

        del(random_context['area'])

        response = client.post("/recommend", json=random_context, headers=headers)

        assert response.status_code == 422

    def test_recommend_with_wrong_api_key(self):
        response = client.post("/recommend", headers={"x-api-key": "wrong"})
        assert response.status_code == 403
