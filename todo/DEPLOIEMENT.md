# Plan de déploiement : API sur Render + dashboard sur Streamlit Community Cloud

## Étape 3 : déployer l'API sur Render
- [ ] **Pendant le build, vérifier dans les logs que le `.joblib` fait bien environ 182 Mo** (ou la nouvelle taille), et pas environ 130 octets. Dans ce second cas, Render a récupéré le fichier « pointeur » LFS au lieu du vrai modèle, et le chargement planterait. Solution de repli : télécharger le modèle au build depuis une **GitHub Release**, avec un `RUN curl -L ... -o models/agriculture_model.joblib` dans le Dockerfile.
- [ ] Tester `https://<nom>.onrender.com/health`, puis `/docs` avec la clé dans le bouton « Authorize ».

## Étape 4 : déployer le dashboard sur Streamlit Community Cloud
- [ ] Aller sur share.streamlit.io (connexion GitHub) → **Create app** → repo, branche `main`, fichier principal `../dashboard/app.py`.
- [ ] **Advanced settings :**
  - **Python :** 3.13 ;
  - **Secrets** (format TOML) :
    ```toml
    API_URL = "https://<nom>.onrender.com"
    API_KEY = "ta-cle"
    ```
    Les secrets de premier niveau sont aussi exposés en variables d'environnement, donc `os.getenv` fonctionne sans changer le code.
- [ ] Tester une prédiction et une recommandation de bout en bout.

## Étape 5 : CI/CD (un plus pour la soutenance)
- [ ] Le déclencher sur `push` et `pull_request` vers `main`.
- [ ] Dans Render (Settings → Auto-Deploy), choisir **« After CI checks pass »** : l'API n'est redéployée que si les tests passent.
- [ ] Streamlit Cloud se redéploie tout seul à chaque push, rien à faire.

Chaîne complète à présenter : push → tests GitHub Actions → déploiement automatique.

## Étape 6 : le jour J
- [ ] **10 min avant :** ouvrir `https://<nom>.onrender.com/health` pour réveiller l'API (veille après 15 min sans requête), puis le dashboard (veille après environ 12 h sans visite).
- [ ] **Plan B :** `docker compose up` en local, plus une courte vidéo de démonstration.
- [ ] L'email apparaît publiquement dans `/docs` (champ `contact` de FastAPI).

---

**Points de vigilance :** la RAM (étape 0) peut tout bloquer, d'où son passage en premier. Ensuite, le seul piège probable est le fichier LFS sur Render (étape 3).
