# Agritech answers

## Mission 

Vous êtes Data Scientist Machine Learning junior dans l’entreprise Agritech Answers, spécialisée dans l’optimisation agricole et l’innovation agrotechnologique.
L'objectif est de développer une application web simple et intuitive pour aider nos clients agriculteurs à prendre de meilleures décisions. 
L'application doit remplir deux fonctions complémentaires, au sein d'une même interface :
- Fonction de prédiction : Permettre à un utilisateur de sélectionner une culture spécifique, de renseigner les conditions de sa parcelle (température, usage de pesticides, etc.) et d'obtenir une estimation chiffrée du rendement attendu. C'est notre moteur d'analyse de base.
- Fonction de recommandation : C'est là que réside notre plus grande valeur ajoutée. L'utilisateur renseigne uniquement les conditions de sa parcelle, et l'application lui recommande la culture la plus rentable en simulant le rendement pour toutes les cultures possibles et en affichant un classement.

Pour réaliser ce projet, nous allons adopter une architecture moderne et découplée :
- Un Back-end : Une API (Application Programming Interface) qui contiendra notre modèle de Machine Learning entraîné. Elle exposera des "endpoints" (des URLs spécifiques) pour la prédiction et la recommandation.
- Un Front-end : Une application web interactive développée avec Streamlit. Cette interface utilisateur sera simple, visuelle, et permettra aux agriculteurs d'interroger notre API sans jamais voir une ligne de code.

Vous serez en charge de l’ensemble du flux de travail, depuis la préparation des données jusqu’à la mise en production d’un prototype fonctionnel.

Les attendus principaux sont les suivants :

- Construction d’un pipeline MLOps et CI/CD robuste, garantissant qualité, reproductibilité et évolutivité du système.
- Fusion et analyse de deux jeux de données distincts :
  - Agriculture CropYield Dataset : Données historiques de rendement selon les cultures et les régions. Une analyse en composantes principales (ACP) est attendue pour identifier les variables clés. Merci de les expliciter clairement.
  - CropYield Prediction Dataset : Données agronomiques et climatiques annuelles (usage de pesticides, fertilisants, température moyenne, etc.) utilisées pour valider l’analyse précédente.

Vous êtes libre de choisir la manière d'exploiter ces deux jeux de données pour construire votre modèle, en fonction de ce qui vous semble le plus pertinent du point de vue analytique et métier. L’important est de justifier clairement vos choix et d’optimiser la performance et l’interprétabilité du modèle.
## Lancer l'application

### Configuration

Le fichier `.env` à la racine doit contenir :

```
API_KEY=<votre clé>             # obligatoire : sans clé, l'API refuse toutes les requêtes (403)
API_URL=http://localhost:8000   # optionnel : adresse de l'API utilisée par le dashboard
```

### Avec Docker (API + dashboard)

```bash
docker compose up --build
```

- API : http://localhost:8000 (documentation : http://localhost:8000/docs)
- Dashboard : http://localhost:8501

### En local

```bash
uv run fastapi run api/main.py --port 8000   # terminal 1 : l'API
uv run streamlit run app.py                  # terminal 2 : le dashboard
```

### Endpoints de l'API

| Méthode | Endpoint | Rôle |
|---|---|---|
| GET | `/health` | Vérifie que l'API répond (sans clé) |
| GET | `/metadata` | Cultures, pays, bornes des variables et performances du modèle (utilisé par le dashboard) |
| POST | `/predict` | Rendement estimé pour une culture et un contexte |
| POST | `/recommend` | Classement de toutes les cultures par rendement estimé |

Toutes les routes sauf `/health` exigent l'en-tête `x-api-key`.

### Le dashboard

Le dashboard (`app.py`) ne contient aucune logique de Machine Learning : il interroge l'API et affiche les résultats.

- **Prédiction** : choix du pays, de la culture et des conditions de la parcelle, puis affichage du rendement estimé en hg/ha.
- **Recommandation** : choix du pays et des conditions, puis affichage de la culture recommandée, d'un graphique à barres et du classement complet.

Chaque variable peut être marquée « Inconnue » : elle est alors envoyée vide (`null`) et le modèle la complète à partir des données historiques.
