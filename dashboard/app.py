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

# Prix de vente indicatifs (€/tonne), modifiables dans le dashboard
PRIX_EUR_TONNE = {
    "Blé": 235.0,
    "Igname": 1100.0,
    "Manioc": 100.0,
    "Maïs": 206.0,
    "Patate douce": 1500.0,
    "Plantains et autres": 3100.0,
    "Pomme de terre": 670.0,
    "Riz": 445.0,
    "Soja": 410.0,
    "Sorgho": 195.0,
}
HG_PAR_TONNE = 10_000  # 10 000 hg/ha = 1 t/ha

# Séparateur de milliers français (espace) dans les axes, infobulles et étiquettes
ALTAIR_LOCALE = {"number": {"decimal": ",", "thousands": "\u202f", "grouping": [3], "currency": ["", " €"]}}

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


def format_eur_ha(value: float) -> str:
    return f"{value:,.0f} €/ha".replace(",", " ")


def ranked_bar_chart(data: pd.DataFrame, field: str, axis_title: str, tooltip: list) -> alt.Chart:
    """Barres horizontales triées par `field` décroissant, la meilleure culture mise en avant."""
    data = data.assign(meilleure=data[field] == data[field].max())
    base = alt.Chart(data).encode(
        y=alt.Y("item:N", sort=alt.EncodingSortField(field=field, order="descending"), title=None),
        x=alt.X(f"{field}:Q", title=axis_title, axis=alt.Axis(format=",.0f")),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
        color=alt.condition(alt.datum.meilleure, alt.value(COLOR_HIGHLIGHT), alt.value(COLOR_OTHERS)),
        tooltip=tooltip,
    )
    # Valeur affichée uniquement sur la meilleure culture, les autres sont dans l'infobulle
    label = base.transform_filter(alt.datum.meilleure).mark_text(align="left", dx=6).encode(
        text=alt.Text(f"{field}:Q", format=",.0f"),
    )
    return (bars + label).properties(height=36 * len(data)).configure(locale=ALTAIR_LOCALE)


def price_inputs(cultures: list[str]) -> dict[str, float]:
    """Tableau éditable des prix de vente, pré-rempli avec les prix par défaut."""
    with st.expander("Prix de vente utilisés (€/tonne) — modifiables"):
        prices = pd.DataFrame(
            {"Culture": cultures, "Prix (€/t)": [PRIX_EUR_TONNE.get(c, 0.0) for c in cultures]}
        )
        edited = st.data_editor(
            prices,
            hide_index=True,
            width="stretch",
            disabled=["Culture"],
            column_config={"Prix (€/t)": st.column_config.NumberColumn(min_value=0.0, step=5.0, format="%.0f €")},
            key="prix_editor",
        )
    return dict(zip(edited["Culture"], edited["Prix (€/t)"].fillna(0.0)))


def show_recommendation(result: list[dict], payload: dict) -> None:
    ranking = pd.DataFrame(result)
    prices = price_inputs(ranking["item"].tolist())
    ranking["yield_t_ha"] = ranking["yield_hg_ha"] / HG_PAR_TONNE
    ranking["price_eur_t"] = ranking["item"].map(prices)
    ranking["revenue_eur_ha"] = ranking["yield_t_ha"] * ranking["price_eur_t"]

    best = ranking.iloc[0]
    best_revenue = ranking.loc[ranking["revenue_eur_ha"].idxmax()]

    st.success(f"Culture recommandée pour **{payload['area']}** : **{best['item']}**")
    col_best, col_second = st.columns(2)
    col_best.metric(f"Meilleur rendement — {best['item']}", format_hg_ha(best["yield_hg_ha"]))
    col_second.metric(f"Meilleur CA estimé — {best_revenue['item']}", format_eur_ha(best_revenue["revenue_eur_ha"]))
    if best_revenue["item"] != best["item"]:
        st.info(
            f"En valeur, **{best_revenue['item']}** rapporte plus que **{best['item']}** "
            "malgré un rendement plus faible, grâce à un prix à la tonne plus élevé."
        )

    st.subheader("Rendement estimé par culture")
    st.altair_chart(
        ranked_bar_chart(
            ranking,
            "yield_hg_ha",
            "Rendement estimé (hg/ha)",
            [
                alt.Tooltip("rank:O", title="Rang"),
                alt.Tooltip("item:N", title="Culture"),
                alt.Tooltip("yield_hg_ha:Q", title="Rendement (hg/ha)", format=",.0f"),
            ],
        ),
        width="stretch",
    )

    st.subheader("Chiffre d'affaires estimé par culture")
    st.altair_chart(
        ranked_bar_chart(
            ranking,
            "revenue_eur_ha",
            "Chiffre d'affaires estimé (€/ha)",
            [
                alt.Tooltip("item:N", title="Culture"),
                alt.Tooltip("yield_t_ha:Q", title="Rendement (t/ha)", format=",.1f"),
                alt.Tooltip("price_eur_t:Q", title="Prix (€/t)", format=",.0f"),
                alt.Tooltip("revenue_eur_ha:Q", title="CA estimé (€/ha)", format=",.0f"),
            ],
        ),
        width="stretch",
    )
    st.caption(
        "CA estimé = rendement estimé (t/ha) × prix de vente (€/t). Prix unique par culture, "
        "indépendant du pays et de l'année ; les coûts de production ne sont pas déduits (ce n'est pas une marge)."
    )


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
