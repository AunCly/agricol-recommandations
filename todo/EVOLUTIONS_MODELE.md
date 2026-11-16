# Évolutions possibles du modèle

Constats faits lors du cycle 5 (`../notebooks/training_cycle5.ipynb`), à traiter plus tard. Le modèle actuellement sauvegardé (`models/agriculture_model.joblib`) est utilisable par l'API en l'état ; ces évolutions amélioreraient sa qualité ou son déploiement.

---

## 1. Ajouter l'année (`Year`) comme variable du modèle (priorité haute)

**Constat** : sur le test (2009 à 2013), le modèle actuel bat à peine la baseline naïve (médiane pays × culture).

| Test 2009-2013 | Baseline | Random Forest actuelle | Évolution |
|---|---|---|---|
| R² | 0.836 | 0.852 | +0.016 |
| RMSE (hg/ha) | 32 646 | 31 012 | −5 % |
| MAE (hg/ha) | 16 461 | 15 808 | −4 % |
| Précision à ±15 % | 47,9 % | 48,5 % | ≈ |

**Cause** : les rendements augmentent avec le temps (progrès agricole), et le modèle n'a aucune variable pour le capter. Il sous-estime le rendement de 9 700 hg/ha en moyenne sur le test.

**Effet mesuré de l'ajout de `Year`** (mêmes hyperparamètres) :

| | Validation croisée (RMSE) | Test (RMSE) | Test (R²) |
|---|---|---|---|
| Baseline | 23 338 | 32 646 | 0.836 |
| RF sans `Year` (actuel) | 24 976 | 31 012 | 0.852 |
| RF avec `Year` | 23 281 | 27 486 (−16 % vs baseline) | 0.884 |

Le gain apparaît déjà en validation croisée (−7 %) : la décision peut se justifier sans utiliser le jeu de test.

**À faire** :
- Ajouter `Year` aux variables numériques dans les cycles 3, 4 et 5, puis relancer et mettre à jour les conclusions. Au minimum, le faire dans le cycle 5 en justifiant le choix par la validation croisée.
- Côté API : ajouter un champ `year`, optionnel, avec l'année en cours par défaut.
- Limite à documenter : au-delà de 2013, le modèle extrapole une tendance qu'il n'a pas observée. Une Random Forest ne prolonge pas une tendance : elle prédit comme pour la dernière année connue.

---

## 2. Corriger les conclusions des cycles 3 et 4 (priorité haute)

Les conclusions des cycles 3 et 4 annoncent un gain d'environ −21 % / −23 % de RMSE par rapport à la baseline. Cette comparaison est fausse : elle oppose la validation croisée du modèle (années 1990-2008) à la baseline évaluée sur le test (2009-2013), qui sont des années différentes.

Calculée sur les **mêmes plis** de validation croisée, la baseline obtient une RMSE de 23 338, contre 24 976 pour la Random Forest optimisée sans `Year`. Le modèle actuel ne fait donc pas mieux que la baseline en validation croisée.

**À faire** :
- Calculer la baseline sur les mêmes plis `TimeSeriesSplit` que les modèles (dans le cycle 0, ou dans chaque cycle), et l'enregistrer dans MLflow.
- Réécrire les passages concernés des conclusions des cycles 3 et 4.

---

## 3. Filtrer ou signaler les cultures absentes du pays dans `/recommend` (priorité moyenne)

**Constat** : pour la France, le modèle classe l'igname, le plantain, la patate douce et le manioc juste derrière la pomme de terre (environ 150 000 hg/ha), alors que ces cultures n'y ont jamais été pratiquées. Le modèle extrapole : « pays à fort rendement » + « culture lourde » donne un gros chiffre, sans réalité agronomique.

**Options** :
- Ne recommander que les cultures déjà pratiquées dans le pays (la liste par pays peut être ajoutée au fichier de métadonnées).
- Ou toutes les renvoyer, avec un indicateur `cultivee_dans_le_pays` (déjà calculé dans le cycle 5).

---

## 4. Rendre le classement de `/recommend` plus pertinent (priorité moyenne)

Trier par rendement brut en hg/ha favorise mécaniquement les tubercules (une pomme de terre pèse bien plus lourd par hectare qu'une céréale). Le cycle 5 calcule un **indice relatif** : le rendement prédit divisé par le rendement médian mondial de la culture. Au-dessus de 1, le contexte est plus favorable que la moyenne pour cette culture.

**À faire** : renvoyer cet indice dans `/recommend` en plus du rendement, et l'expliquer dans le rapport. Pour parler de profitabilité réelle, il faudrait ajouter des données de prix par culture (par exemple, les prix à la production de la FAO).

---

## 5. Réduire la taille du modèle (priorité moyenne)

**Constat** : le fichier fait 720 Mo, et environ 180 Mo compressé (`compress=3`, déjà en place). C'est au-dessus de la limite GitHub (100 Mo par fichier) et lourd pour une image Docker. La taille vient des arbres très profonds retenus au cycle 4 (`max_depth=40`, `min_samples_leaf=1`, `bootstrap=False`).

**Options** :
- Ne pas versionner le fichier : l'ajouter au `../.gitignore` et le régénérer avec le notebook du cycle 5. Ou utiliser Git LFS.
- Alléger le modèle : `min_samples_leaf` à 2 ou 5, `max_depth` à 20, ou moins d'arbres. Vérifier la perte en validation croisée, elle est souvent négligeable.

---

## 6. Points techniques mineurs

- **`TargetEncoder(random_state=42)`** est déprécié dans scikit-learn 1.9 (avertissements `FutureWarning`). Utiliser `TargetEncoder(target_type='continuous', cv=KFold(5, shuffle=True, random_state=42))`, déjà fait dans le cycle 5. À reporter dans les cycles 1 à 4, et à ajouter là où il n'y a encore aucune graine (cycles 1 à 3).
- **Importer la classe `AgritechFeatureEngineering`** via `notebooks.agritech_feature_engineering`, comme dans le cycle 5, dans tout notebook qui sauvegarde un modèle : l'API doit pouvoir la ré-importer depuis la racine du projet. À terme, déplacer ce code dans un vrai package (par exemple `src/`), indépendant des notebooks.
- **Sous-estimation systématique** : la cible en log suivie du retour en hg/ha (`expm1`) tend à sous-estimer les rendements. Une correction simple consiste à multiplier les prédictions par un facteur calculé sur le train.
- **Nettoyage MLflow** avant les captures : les runs `FAILED` du cycle 4, les runs en double, et le dossier `../notebooks/mlruns` créé par un test.
