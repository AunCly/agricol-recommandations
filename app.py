"""Dashboard Streamlit : front-end de l'API de prédiction de rendement agricole.

Aucune logique de Machine Learning ici : l'application interroge l'API FastAPI
et affiche ses réponses.
"""
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
TIMEOUT_SECONDS = 30

MODE_PREDICTION = "Prédiction"
MODE_RECOMMANDATION = "Recommandation"

# Couleurs du graphique : culture recommandée mise en avant, les autres en retrait
COLOR_HIGHLIGHT = "#2a78d6"
COLOR_OTHERS = "#9ec5f4"

# Libellés et pas des variables du contexte (les bornes viennent de l'API)
FEATURES = {
    "avg_temp": {"label": "Température moyenne (°C)", "step": 0.1, "widget": "slider"},
    "average_rain_fall_mm_per_year": {"label": "Précipitations (mm/an)", "step": 10.0, "widget": "slider"},
    "pesticides_kg_ha": {"label": "Pesticides (kg/ha)", "step": 0.1, "widget": "slider"},
    # Distribution très étalée (0 à ~470 000 t) : un champ de saisie est plus pratique qu'un slider
    "pesticides_tonnes": {"label": "Pesticides utilisés dans le pays (tonnes)", "step": 100.0, "widget": "number"},
}


class ApiError(Exception):
    """Erreur lisible renvoyée par l'appel à l'API."""


def call_api(method: str, endpoint: str, payload: dict | None = None) -> dict | list:
    """Appelle l'API et transforme les erreurs techniques en messages compréhensibles."""
    try:
        response = requests.request(
            method,
            f"{API_URL}{endpoint}",
            json=payload,
            headers={"x-api-key": API_KEY},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.ConnectionError:
        raise ApiError(f"Impossible de joindre l'API ({API_URL}). Est-elle démarrée ?")
    except requests.Timeout:
        raise ApiError(f"L'API n'a pas répondu en moins de {TIMEOUT_SECONDS} secondes.")

    if response.status_code == 403:
        raise ApiError("Clé API refusée : vérifiez la variable API_KEY dans le fichier .env.")
    if response.status_code == 422:
        raise ApiError(f"Données refusées par l'API : {response.json().get('detail')}")
    if not response.ok:
        raise ApiError(f"Erreur de l'API (code {response.status_code}).")

    return response.json()


def api_is_alive() -> bool:
    try:
        requests.get(f"{API_URL}/health", timeout=5).raise_for_status()
        return True
    except requests.RequestException:
        return False


@st.cache_data(ttl=300, show_spinner="Chargement des cultures et des pays…")
def get_metadata() -> dict:
    return call_api("GET", "/metadata")


def context_inputs(metadata: dict) -> dict:
    """Affiche les champs du contexte et renvoie les valeurs (None si inconnue)."""
    st.subheader("Données")

    values = {}
    columns = st.columns(2)
    for index, (name, config) in enumerate(FEATURES.items()):
        bounds = metadata["feature_ranges"][name]
        with columns[index % 2]:
            unknown = st.checkbox("Inconnue", key=f"unknown_{name}")
            common = {
                "label": config["label"],
                "min_value": float(bounds["min"]),
                "max_value": float(bounds["max"]),
                "value": float(bounds["median"]),
                "step": config["step"],
                "disabled": unknown,
                "key": name,
            }
            if config["widget"] == "slider":
                value = st.slider(**common)
            else:
                value = st.number_input(**common)
            values[name] = None if unknown else value
    return values


def format_hg_ha(value: float) -> str:
    """Formate un rendement en hg/ha avec une espace comme séparateur de milliers."""
    return f"{value:,.0f} hg/ha".replace(",", " ")


def show_prediction(result: dict, payload: dict) -> None:
    st.metric(
        label=f"Rendement estimé — {payload['item']} ({payload['area']})",
        value=format_hg_ha(result["yield_hg_ha"]),
    )


def show_recommendation(result: list[dict], payload: dict) -> None:
    ranking = pd.DataFrame(result)
    best = ranking.iloc[0]

    st.success(f"Culture recommandée pour **{payload['area']}** : **{best['item']}**")
    col_best, col_second = st.columns(2)
    col_best.metric("Rendement estimé (n°1)", format_hg_ha(best["yield_hg_ha"]))

    st.subheader("Rendement estimé par culture")
    chart_data = ranking.assign(recommandee=ranking["rank"] == 1)
    base = alt.Chart(chart_data).encode(
        y=alt.Y("item:N", sort=alt.SortField("rank"), title=None),
        x=alt.X("yield_hg_ha:Q", title="Rendement estimé (hg/ha)", axis=alt.Axis(format=",.0f")),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
        color=alt.condition(
            alt.datum.recommandee,
            alt.value(COLOR_HIGHLIGHT),
            alt.value(COLOR_OTHERS),
        ),
        tooltip=[
            alt.Tooltip("rank:O", title="Rang"),
            alt.Tooltip("item:N", title="Culture"),
            alt.Tooltip("yield_hg_ha:Q", title="Rendement (hg/ha)", format=",.0f"),
        ],
    )
    # Valeur affichée uniquement sur la culture recommandée, les autres sont dans le tableau
    label = base.transform_filter(alt.datum.recommandee).mark_text(align="left", dx=6).encode(
        text=alt.Text("yield_hg_ha:Q", format=",.0f"),
    )
    chart = (bars + label).properties(height=36 * len(ranking)).configure(
        # Séparateur de milliers français (espace) dans l'axe, les infobulles et l'étiquette
        locale={"number": {"decimal": ",", "thousands": "\u202f", "grouping": [3], "currency": ["", " €"]}}
    )
    st.altair_chart(chart, width="stretch")

def main() -> None:
    st.set_page_config(page_title="Agritech Answers", page_icon="🌾", layout="centered")
    st.title("Agritech")

    with st.sidebar:
        st.header("Mode")
        mode = st.radio(
            "Que voulez-vous faire ?",
            [MODE_PREDICTION, MODE_RECOMMANDATION],
            captions=[
                "Estimer le rendement d'une culture choisie",
                "Classer toutes les cultures par rendement estimé",
            ],
        )

    try:
        metadata = get_metadata()
    except ApiError as error:
        st.error(str(error))
        st.stop()

    st.subheader("Localisation et culture")
    pays = metadata["pays"]
    area = st.selectbox("Pays", pays, index=pays.index("France") if "France" in pays else 0)
    payload = {"area": area}
    if mode == MODE_PREDICTION:
        payload["item"] = st.selectbox("Culture", metadata["cultures"])

    payload.update(context_inputs(metadata))

    button_label = "Estimer" if mode == MODE_PREDICTION else "Recommander"
    if st.button(button_label, type="primary", width="stretch"):
        endpoint = "/predict" if mode == MODE_PREDICTION else "/recommend"
        try:
            with st.spinner("Interrogation de l'API…"):
                result = call_api("POST", endpoint, payload)
            st.session_state["last"] = {"mode": mode, "payload": payload, "result": result}
        except ApiError as error:
            st.error(str(error))

    # Le dernier résultat reste affiché tant que l'on reste dans le même mode
    last = st.session_state.get("last")
    if last and last["mode"] == mode:
        st.divider()
        if last["payload"] != payload:
            st.info("Les paramètres ont changé depuis ce résultat : relancez la requête pour le mettre à jour.")
        if mode == MODE_PREDICTION:
            show_prediction(last["result"], last["payload"])
        else:
            show_recommendation(last["result"], last["payload"])

main()
