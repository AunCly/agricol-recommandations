# Agritech Answers : prédiction de rendement et recommandation de cultures

Application web qui aide les agriculteurs à décider quoi planter :

- **Prédiction** : pour un pays, une culture et des conditions de parcelle (température, pluie, pesticides), l'application estime le rendement attendu.
- **Recommandation** : à partir des seules conditions de la parcelle, l'application simule le rendement de toutes les cultures et les classe, du meilleur au moins bon.

Le projet couvre toute la chaîne : préparation des données, entraînement suivi dans MLflow, API FastAPI qui sert le modèle, dashboard Streamlit, tests et CI/CD.

---

## Sommaire

1. [Contexte de la mission](#1-contexte-de-la-mission)
2. [Architecture](#2-architecture)
3. [Arborescence du projet](#3-arborescence-du-projet)
4. [Données](#4-données)
5. [Modélisation](#5-modélisation)
6. [Artefacts du modèle](#6-artefacts-du-modèle)
7. [API](#7-api)
8. [Dashboard](#8-dashboard)
9. [Installation et lancement](#9-installation-et-lancement)
10. [Tests et CI/CD](#10-tests-et-cicd)
11. [Déploiement](#11-déploiement)
12. [Limites connues et évolutions](#12-limites-connues-et-évolutions)

---

## 1. Contexte de la mission

Projet réalisé en tant que Data Scientist Machine Learning junior chez Agritech Answers, entreprise spécialisée dans l'optimisation agricole.

Attendus :

- un pipeline MLOps et CI/CD qui garantit la qualité, la reproductibilité et l'évolutivité du système ;
- la fusion et l'analyse de deux jeux de données :
  - **Agriculture CropYield Dataset** : rendements historiques par culture et par pays ;
  - **CropYield Prediction Dataset** : données agronomiques et climatiques annuelles (pesticides, température moyenne, précipitations) ;
- une architecture découplée : un **back-end** (API qui contient le modèle) et un **front-end** Streamlit qui interroge l'API sans exposer de code à l'utilisateur.

---

## 2. Architecture

```
┌───────────────────────┐   HTTP + x-api-key   ┌──────────────────────────────┐
│  Dashboard Streamlit  │ ───────────────────▶ │  API FastAPI (api/main.py)   │
│  dashboard/app.py     │ ◀─────────────────── │                              │
│  (aucune logique ML)  │        JSON          │  AgricultureYieldPredictor   │
└───────────────────────┘                      │  (models/predictor.py)       │
                                               │          │                   │
                                               │          ▼                   │
                                               │  models/agriculture_model    │
                                               │    .joblib + _metadata.json  │
                                               └──────────────────────────────┘
                                                          ▲
                                                          │ sauvegarde
┌──────────────────────────────────────────────────────────┴──────────────────┐
│  Notebooks : exploration → processing → analyse → cycles d'entraînement     │
│  Suivi des expériences dans MLflow (mlflow.db, mlartifacts/)                │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Brique | Technologie |
|---|---|
| Langage | Python 3.13 |
| Gestion des dépendances | [uv](https://docs.astral.sh/uv/) (`pyproject.toml`, `uv.lock`) |
| Données et modèle | pandas, NumPy, scikit-learn |
| Suivi des expériences | MLflow |
| API | FastAPI, Pydantic |
| Front-end | Streamlit, Altair |
| Conteneurisation | Docker, Docker Compose |
| Tests | pytest, pytest-cov |
| CI/CD | GitHub Actions |
| Fichiers volumineux | Git LFS (`*.csv`, `*.joblib`, `*.json`, `*.png`) |

---

## 3. Arborescence du projet

```
.
├── api/
│   ├── main.py                  # Application FastAPI : routes et authentification
│   └── agricol_context.py       # Schémas Pydantic des requêtes
├── models/
│   ├── predictor.py             # Chargement du modèle, prédiction et recommandation
│   ├── agriculture_model.joblib # Modèle de production (Git LFS)
│   └── agriculture_model_metadata.json  # Métadonnées lues par l'API
├── pipeline/
│   └── train.py                 # Réentraînement : entraînement, évaluation, validation, sauvegarde
├── dashboard/
│   ├── app.py                   # Interface Streamlit
│   └── requirements.txt         # Dépendances pour Streamlit Community Cloud
├── notebooks/
│   ├── exploration.ipynb        # Découverte des fichiers bruts
│   ├── processing.ipynb         # Nettoyage, harmonisation et fusion
│   ├── analysis.ipynb           # Analyse exploratoire
│   ├── training_baseline.ipynb  # Cycle 0 : baseline naïve
│   ├── training_cycle1..6.ipynb # Cycles d'entraînement successifs
│   ├── agritech_feature_engineering.py  # Transformer scikit-learn (variables dérivées)
│   └── utils.py                 # Métriques et découpage temporel train/test
├── data/
│   ├── raw/                     # Fichiers sources
│   ├── processed/               # Fichiers nettoyés, un par source
│   └── rafined/                 # Jeu de données final fusionné
├── tests/
│   ├── test_api.py              # Tests de l'API
│   └── test_pipeline.py         # Tests du pipeline (validation, entraînement sur échantillon)
├── todo/                        # Rapport métier, plan de présentation, évolutions
├── .github/workflows/           # Pipelines CI/CD
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml / uv.lock
└── .env.example
```

---

## 4. Données

### 4.1 Sources (`data/raw/`)

| Fichier | Contenu |
|---|---|
| `yield.csv`, `yield_df.csv`, `crop_yield.csv` | Rendements FAO (hg/ha) par pays, culture et année |
| `pesticides.csv`, `Inputs_Pesticides_Use_E_All_Data_NOFLAG.csv` | Usage de pesticides (FAO) |
| `rainfall.csv`, `average-precipitation-per-year.csv` | Précipitations annuelles moyennes |
| `temp.csv`, `combined_temperature 2.csv` | Températures moyennes |
| `wikipedia-iso-country-codes.csv`, `Production_Crops_Livestock_E_AreaCodes.csv` | Codes pays (ISO et FAO), utilisés pour les jointures |

### 4.2 Traitement

1. **`exploration.ipynb`** : description de chaque fichier, repérage des clés de jointure (Iso Code, Area Code).
2. **`processing.ipynb`** :
   - suppression des agrégats géographiques en double (régions, continents) ;
   - harmonisation des noms de pays, notamment dans le fichier de températures ;
   - reprise des pesticides depuis le fichier FAO brut ;
   - fusion de toutes les sources par pays et par année ;
   - écriture du résultat dans `data/processed/` puis `data/rafined/`.
3. **`analysis.ipynb`** : analyse de la cible et des variables explicatives, focus sur la France.

### 4.3 Jeu de données final (`data/rafined/dataset.csv`)

**25 848 lignes**, **211 pays**, **10 cultures**, de **1990 à 2013**. Le fichier `dataset_1960_2013.csv` conserve une version plus longue, sans les variables climatiques complètes.

| Colonne | Type | Description |
|---|---|---|
| `Area` | texte | Pays |
| `Item` | texte | Culture (Blé, Igname, Manioc, Maïs, Patate douce, Plantains et autres, Pomme de terre, Riz, Soja, Sorgho) |
| `Year` | entier | Année |
| `hg/ha` | entier | **Cible** : rendement en hectogrammes par hectare (10 000 hg/ha = 1 t/ha) |
| `average_rain_fall_mm_per_year` | réel | Précipitations annuelles moyennes (mm) |
| `pesticides_tonnes` | réel | Pesticides utilisés dans le pays (tonnes) |
| `pesticides_kg_ha` | réel | Pesticides rapportés à la surface (kg/ha) |
| `avg_temp` | réel | Température moyenne annuelle (°C) |
| `missing_pesticide_value`, `missing_rain_value`, `missing_temp_value` | 0/1 | Indicateurs de valeur manquante |
| `Area ISO Code`, `Area ISO Alpha-2 Code`, `Area ISO Alpha-3 Code`, `Code` | — | Codes pays, non utilisés par le modèle |

Points d'attention :

- Les variables climatiques et les pesticides sont **nationales et annuelles** : une valeur par pays et par an, pas par parcelle.
- Environ **28 % des températures sont manquantes** (74 pays sans aucune valeur). Les valeurs manquantes sont conservées, puis imputées dans le pipeline. Les indicateurs `missing_*` signalent au modèle qu'elles étaient absentes.

---

## 5. Modélisation

### 5.1 Protocole

- **Découpage temporel** (`notebooks/utils.py::test_train_time_split`) : l'année de césure est la première année où le cumul des lignes dépasse 80 %. Résultat : **train 1990-2008**, **test 2009-2013**. Le modèle est évalué sur des années qu'il n'a jamais vues, comme en conditions réelles.
- **Validation croisée** : `TimeSeriesSplit(5)` sur le train uniquement, train trié par année. Le test n'est utilisé qu'une fois, pour l'évaluation finale.
- **Métriques** : R², MAE, RMSE (en hg/ha), et **précision à ±15 %** (part des prédictions à moins de 15 % du rendement réel).
- **Suivi** : chaque cycle enregistre paramètres, métriques et artefacts dans l'expérience MLflow `Agriculture_optimization`.

### 5.2 Cycles d'entraînement

| Cycle | Notebook | Contenu |
|---|---|---|
| 0 | `training_baseline.ipynb` | Baseline naïve : médiane historique du couple pays × culture |
| 1 | `training_cycle1.ipynb` | Régression linéaire |
| 2 | `training_cycle2.ipynb` | Régression linéaire sur cible transformée (`log1p`) |
| 3 | `training_cycle3.ipynb` | Comparaison de plusieurs familles de modèles |
| 4 | `training_cycle4.ipynb` | Optimisation des hyperparamètres de la Random Forest |
| 5 | `training_cycle5.ipynb` | Évaluation finale sur le test, importance des variables, réentraînement sur 1990-2013 et sauvegarde pour l'API |
| 6 | `training_cycle6.ipynb` | Allègement du modèle pour tenir dans 512 Mo de RAM (offre gratuite de Render) |

### 5.3 Pipeline du modèle

```
TransformedTargetRegressor(func=log1p, inverse_func=expm1)
└── Pipeline
    ├── features      AgritechFeatureEngineering
    │                   aridity         = (avg_temp + 10) / (average_rain_fall_mm_per_year + 1)
    │                   avg_temp_square = avg_temp²
    ├── preprocessor  ColumnTransformer
    │                   Area      → TargetEncoder (continu, KFold 5)
    │                   Item      → OneHotEncoder (handle_unknown='ignore')
    │                   numériques → SimpleImputer(median) → StandardScaler
    └── model         RandomForestRegressor
```

Le modèle de production a été entraîné au cycle 5 :

| Hyperparamètre | Valeur |
|---|---|
| `n_estimators` | 200 |
| `max_features` | 0.5 |
| `max_depth` | 40 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 1 |
| `bootstrap` | False |
| `random_state` | 42 |

> **Import du transformer** : le `.joblib` référence la classe `notebooks.agritech_feature_engineering.AgritechFeatureEngineering`. L'API doit donc tourner depuis la racine du projet pour que l'import fonctionne.

### 5.4 Performances (test 2009-2013)

| Indicateur | Baseline pays × culture | Random Forest |
|---|---|---|
| R² | 0.836 | **0.852** |
| MAE (hg/ha) | 16 461 | **15 808** |
| RMSE (hg/ha) | 32 646 | **31 012** |
| Précision à ±15 % | 47,9 % | **48,5 %** |

Le couple pays × culture explique déjà l'essentiel du rendement. Le modèle ne fait que légèrement mieux que la baseline (environ −5 % de RMSE) et sous-estime les rendements récents, faute de variable temporelle (voir [section 12](#12-limites-connues-et-évolutions)).

Importances des variables : culture ≈ 55 %, pays ≈ 20 %, climat ≈ 13 %, pesticides ≈ 11 %, indicateurs de valeurs manquantes ≈ 1 %.

### 5.5 Reproduire l'entraînement

```bash
# 1. Démarrer le serveur MLflow (les notebooks pointent vers http://127.0.0.1:5000)
uv run mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts

# 2. Exécuter les notebooks dans l'ordre : processing → cycles 0 à 5 (ou 6)
uv run jupyter lab   # ou l'IDE, avec le noyau du .venv
```

Les cycles 5 et 6 réécrivent `models/agriculture_model.joblib` et `models/agriculture_model_metadata.json` ; l'API les charge sans modification de code.

---

## 6. Artefacts du modèle

| Fichier | Rôle |
|---|---|
| `models/agriculture_model.joblib` | Pipeline scikit-learn complet, compressé (`compress=3`) : environ 182 Mo sur disque, environ 950 Mo une fois chargé en mémoire |
| `models/agriculture_model_metadata.json` | Version, date, version de scikit-learn, hyperparamètres, variables d'entrée et types, liste des cultures et des pays, bornes min/médiane/max des variables, années d'entraînement, performances sur le test |
| `models/metadata_*.json` | Métadonnées d'anciennes versions (historique) |

Le fichier de métadonnées alimente l'endpoint `/metadata`. Le dashboard en tire les listes déroulantes et les bornes des curseurs.

Les deux fichiers sont versionnés avec **Git LFS**. Après un clone, lancez `git lfs pull` : sans cela, le `.joblib` n'est qu'un pointeur de quelques octets et l'API ne démarre pas.

---

## 7. API

Code : `api/main.py`. Documentation interactive : `http://localhost:8000/docs`.

### 7.1 Authentification

Toutes les routes, sauf `/health`, exigent l'en-tête `x-api-key`. Sa valeur doit être égale à la variable d'environnement `API_KEY`. Si la clé est absente ou fausse, ou si `API_KEY` n'est pas définie côté serveur, l'API répond **403**.

### 7.2 Endpoints

| Méthode | Endpoint | Clé | Rôle |
|---|---|---|---|
| GET | `/health` | non | Vérifie que l'API répond : `{"message": "Alive !"}` |
| GET | `/metadata` | oui | Version, cultures, pays, bornes des variables, années d'entraînement, performances |
| POST | `/predict` | oui | Rendement estimé pour une culture dans un contexte |
| POST | `/recommend` | oui | Classement de toutes les cultures par rendement estimé décroissant |

### 7.3 Schémas des requêtes (`api/agricol_context.py`)

| Champ | Type | `/predict` | `/recommend` |
|---|---|---|---|
| `area` | str | obligatoire | obligatoire |
| `item` | str | obligatoire | — |
| `avg_temp` | float \| null | optionnel | optionnel |
| `average_rain_fall_mm_per_year` | float \| null | optionnel | optionnel |
| `pesticides_tonnes` | float \| null | optionnel | optionnel |
| `pesticides_kg_ha` | float \| null | optionnel | optionnel |

Une variable absente ou `null` est imputée par la médiane du pipeline, et l'indicateur `missing_*` correspondant passe à 1 (`models/predictor.py::build_row`).

### 7.4 Exemples

```bash
curl -X POST http://localhost:8000/predict \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"area": "France", "item": "Blé", "avg_temp": 11.5,
       "average_rain_fall_mm_per_year": 867, "pesticides_tonnes": 62000, "pesticides_kg_ha": 3.4}'
```

```json
{"yield_hg_ha": 71234.5, "yield_t_ha": 7.12}
```

```bash
curl -X POST http://localhost:8000/recommend \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"area": "France", "avg_temp": 11.5}'
```

```json
[
  {"rank": 1, "item": "Pomme de terre", "yield_hg_ha": 412345.0, "yield_t_ha": 41.2},
  {"rank": 2, "item": "...", "yield_hg_ha": 0, "yield_t_ha": 0}
]
```

*Valeurs données à titre d'illustration.*

### 7.5 Fonctionnement interne (`models/predictor.py`)

- `AgricultureYieldPredictor` charge le modèle et ses métadonnées une seule fois, au démarrage de l'API.
- `predict` construit une ligne d'entrée, avec les colonnes dans l'ordre de l'entraînement, et renvoie le rendement en hg/ha et en t/ha.
- `recommend` construit une ligne par culture connue, fait **une seule** prédiction groupée, trie les résultats par rendement décroissant et ajoute un rang.

---

## 8. Dashboard

Code : `dashboard/app.py`. Le dashboard n'appelle que l'API ; il ne contient aucune logique de Machine Learning.

- **Mode Prédiction** : choix du pays, de la culture et des conditions, puis affichage du rendement estimé.
- **Mode Recommandation** : choix du pays et des conditions, puis :
  - culture au meilleur rendement et graphique du rendement par culture ;
  - **chiffre d'affaires estimé** (rendement en t/ha × prix en €/t) avec un second classement. Les prix par défaut (constante `PRIX_EUR_TONNE`) sont modifiables dans un tableau. Le CA ne déduit aucun coût de production : ce n'est pas une marge.
- Chaque variable peut être marquée « Inconnue » : elle est alors envoyée à `null`.
- Les bornes des curseurs viennent de `/metadata`. Un message prévient quand les paramètres ont changé depuis le dernier résultat.

Variables d'environnement lues : `API_URL` (par défaut `http://localhost:8000`) et `API_KEY`.

---

## 9. Installation et lancement

### 9.1 Prérequis

- Python 3.13 et [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Git LFS](https://git-lfs.com/)
- Docker et Docker Compose (facultatif)

```bash
git clone <url-du-depot>
cd <dossier>
git lfs pull     # récupère les données et le modèle
uv sync          # crée le .venv et installe les dépendances
```

### 9.2 Configuration

Copiez `.env.example` en `.env` à la racine, puis complétez-le :

```
API_KEY=<votre clé>             # obligatoire : sans clé, l'API refuse toutes les requêtes (403)
API_URL=http://localhost:8000   # optionnel : adresse de l'API utilisée par le dashboard
```

### 9.3 En local

```bash
uv run fastapi run api/main.py --port 8000   # terminal 1 : l'API
uv run streamlit run dashboard/app.py        # terminal 2 : le dashboard
```

### 9.4 Avec Docker Compose

```bash
docker compose up --build
```

| Service | URL | Détail |
|---|---|---|
| `api` | http://localhost:8000 (docs : `/docs`) | Port 7860 dans le conteneur |
| `streamlit` | http://localhost:8501 | Joint l'API via `API_URL=http://api:7860` |

Les deux services utilisent la même image (`Dockerfile`), et le code est monté en volume. Le `CMD` par défaut lance FastAPI sur `$PORT` (7860 par défaut, ou la valeur fournie par Render).

---

## 10. Tests et CI/CD

### 10.1 Tests

`tests/test_api.py` teste l'API avec le `TestClient` de FastAPI, sur des contextes tirés au hasard dans `data/rafined/dataset.csv` :

- `/health` répond sans clé ;
- une mauvaise clé renvoie 403 (`/predict`, `/recommend`) ;
- `/metadata` renvoie les champs attendus ;
- `/predict` et `/recommend` répondent, y compris quand des variables manquent.

```bash
uv run pytest                                      # tous les tests
uv run pytest --cov=api --cov=models --cov-report=html   # avec couverture (rapport dans htmlcov/)
```

Les tests lisent `API_KEY` dans le `.env` ou dans l'environnement.

### 10.2 GitHub Actions

| Workflow | Déclencheur | Étapes |
|---|---|---|
| `.github/workflows/test.yml` | push sur `develop` | checkout avec LFS → uv → Python 3.13 → `uv sync` → `pytest` |
| `.github/workflows/test_and_deploy.yml` | push sur `main` | mêmes étapes, préalable au déploiement |
| `.github/workflows/retrain.yml` | push sur `main` modifiant `data/rafined/dataset.csv` ou `pipeline/`, ou lancement manuel | réentraînement → évaluation → validation → tests → commit du nouveau modèle |

La clé de test est fournie par le secret de dépôt `API_KEY`.

Chaîne visée : **push → tests GitHub Actions → redéploiement automatique** (Render en mode *Auto-Deploy : After CI checks pass*).

### 10.3 Pipeline de réentraînement

Le script `pipeline/train.py` reprend le modèle retenu au cycle 6 et automatise son cycle de vie :

```bash
uv run python -m pipeline.train
```

| Étape | Ce qui est fait |
|---|---|
| 1. Données | lecture de `data/rafined/dataset.csv`, découpage temporel train (avant 2009) / test (2009-2013) |
| 2. Entraînement | Random Forest du cycle 6 entraînée sur le train |
| 3. Évaluation | R², MAE et RMSE sur le test, comparés à la baseline naïve (médiane pays × culture) |
| 4. Validation | le modèle est accepté s'il **bat la baseline** et si sa RMSE ne dépasse pas **de plus de 10 %** celle du modèle en production |
| 5. Sauvegarde | si accepté : réentraînement sur toutes les années, écriture du `.joblib` et des métadonnées (scores inclus) |

Chaque exécution est tracée dans MLflow (run « Pipeline - réentraînement »), avec la décision d'acceptation en tag. Si le modèle est refusé, rien n'est écrit et le script se termine en erreur.

Dans GitHub Actions (`retrain.yml`), le nouveau modèle n'est commité sur `main` que si la validation **et** les tests de l'API passent ; ce commit déclenche le redéploiement sur Render. Le rapport d'évaluation (métadonnées) est conservé en artefact du workflow.

Limite connue : le modèle en production est entraîné sur toutes les années, il ne peut donc pas être réévalué honnêtement sur le test. La comparaison se fait avec la RMSE enregistrée dans ses métadonnées.

---

## 11. Déploiement

| Composant | Hébergeur | Configuration |
|---|---|---|
| API | Render (Docker) | Variable `API_KEY` ; Render fournit `$PORT`. Vérifier dans les logs de build que le `.joblib` fait bien environ 182 Mo, et non quelques octets (pointeur LFS non résolu). |
| Dashboard | Streamlit Community Cloud | Fichier principal `dashboard/app.py`, Python 3.13 ; secrets `API_URL` (URL Render) et `API_KEY` |

Points de vigilance :

- **Mémoire** : le modèle du cycle 5 occupe environ 950 Mo en RAM, au-delà des 512 Mo de l'offre gratuite de Render. Le cycle 6 sélectionne une variante plus légère (budget de 250 Mo pour le modèle) ; il faut l'exécuter et sauvegarder son modèle avant le déploiement.
- **Mise en veille** : l'offre gratuite de Render endort l'API après 15 minutes sans requête. Appelez `/health` avant une démonstration.
- Le champ `contact` de FastAPI est visible publiquement dans `/docs`.

---

## 12. Limites connues et évolutions

- **Pas de variable temporelle** : les rendements progressent dans le temps, et le modèle les sous-estime d'environ 1 t/ha sur 2009-2013. L'ajout de `Year` réduit la RMSE de test d'environ 16 % par rapport à la baseline.
- **Cultures hors contexte** : `/recommend` peut classer haut des cultures jamais pratiquées dans le pays (par exemple, l'igname en France).
- **Classement en hg/ha** : il favorise mécaniquement les tubercules. Un indice relatif (rendement prédit / médiane mondiale de la culture) ou des prix par pays rendraient le classement plus pertinent.
- **Données nationales** : le climat et les pesticides décrivent un pays, pas une parcelle. Aucune donnée sur les engrais, l'irrigation ou le sol.
- **ACP** : l'analyse en composantes principales demandée par le cahier des charges reste à formaliser dans les notebooks.
- **Packaging** : le transformer `AgritechFeatureEngineering` est encore dans `notebooks/` ; il faudrait le déplacer dans un package dédié (par exemple `src/`).

Détail et priorités : [`todo/EVOLUTIONS_MODELE.md`](todo/EVOLUTIONS_MODELE.md). Synthèse pour les métiers : [`todo/RAPPORT_METIER.md`](todo/RAPPORT_METIER.md).
