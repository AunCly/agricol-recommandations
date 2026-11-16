# Rapport de synthèse métier : prédiction du rendement et recommandation de cultures

> **Brouillon** du 11/11/2026. Les passages marqués **[À COMPLÉTER]** attendent un résultat qui n'est pas encore produit ou pas encore sauvegardé dans les notebooks.

**Destinataires** : direction, équipes conseil agronomique, chefs de produit Agritech Answers
**Auteur** : Aurélien Clugery, Data Scientist

---

## 1. En bref

- **Ce que fait l'outil** : à partir du pays et des conditions d'une parcelle (température, pluviométrie, pesticides), il estime le rendement d'une culture (fonction *Prédiction*) et classe les 10 cultures étudiées (fonction *Recommandation*).
- **Fiabilité** : sur des années que le modèle n'a jamais vues (2009 à 2013), il explique **85 % des écarts de rendement** (R² = 0,85). Son erreur moyenne est de **1,6 t/ha**, et **une prévision sur deux** tombe à ±15 % du rendement réel.
- **Ce qui détermine le rendement** : d'abord **la culture choisie** (55 % de l'importance dans le modèle), puis **le pays** (20 %), qui résume les sols, la mécanisation et l'irrigation. Le climat et les pesticides pèsent ensemble environ **un quart**.
- **Limite principale** : l'outil fait à peine mieux qu'une règle simple (« le rendement habituel de cette culture dans ce pays »). Il apporte surtout une **estimation cohérente et automatisée** ; une évolution identifiée (prise en compte de l'année) réduirait l'erreur de 16 % (voir section 6).
- **Rendement ≠ profit** : l'outil classe aujourd'hui les cultures en tonnes par hectare. Sans prix de vente ni coûts de production, il ne peut pas encore désigner la culture **la plus rentable**.

---

## 2. Contexte et objectif

Agritech Answers veut aider ses clients agriculteurs à décider **quoi planter** et **à quel rendement s'attendre**. Le prototype livré comprend :

- une **API** qui contient le modèle et expose les fonctions de prédiction et de recommandation ;
- un **tableau de bord** (Streamlit) qui permet de l'utiliser sans aucune compétence technique.

Ce rapport présente ce que les données nous apprennent, la fiabilité du modèle, et nos recommandations pour optimiser rendement et profit.

---

## 3. Les données utilisées

| Source | Contenu | Rôle |
|---|---|---|
| FAO : rendements agricoles (*Agriculture CropYield*) | Rendement en hg/ha par pays, culture et année | Variable à prédire |
| *CropYield Prediction Dataset*, complété par d'autres sources (FAO pesticides, précipitations annuelles, températures) | Pesticides, pluviométrie et température moyenne par pays et par année | Variables explicatives |

**Périmètre final** : 25 848 observations, **211 pays**, **10 cultures**, de **1990 à 2013**.

**Unité** : le rendement est exprimé en hectogrammes par hectare. **10 000 hg/ha = 1 tonne par hectare.**

**Points de vigilance sur les données** :
- Les données climatiques et de pesticides sont **nationales et annuelles** : une seule valeur par pays et par an. Elles ne décrivent pas une parcelle précise.
- Environ **28 % des températures sont manquantes** (74 pays n'ont aucune valeur). Le modèle les remplace par une valeur typique et sait qu'elles étaient absentes.
- Aucune information sur les **engrais**, l'**irrigation**, le **type de sol** ou les **prix** n'est disponible.

---

## 4. Ce que disent les données

### 4.1 La culture fixe l'ordre de grandeur du rendement

| Culture | Rendement médian (t/ha) | Évolution 1990-94 → 2009-13 |
|---|---|---|
| Pomme de terre | 15,1 | +25 % |
| Manioc | 9,2 | +18 % |
| Plantains et autres | 7,9 | +18 % |
| Igname | 7,7 | +16 % |
| Patate douce | 7,5 | +34 % |
| Riz | 3,0 | +36 % |
| Blé | 2,3 | +28 % |
| Maïs | 2,3 | **+75 %** |
| Soja | 1,5 | +19 % |
| Sorgho | 1,2 | +21 % |

- Les **tubercules et plantains** produisent **5 à 10 fois plus de tonnes par hectare** que les céréales et le soja. Ce n'est pas une anomalie : ces plantes sont naturellement plus lourdes et plus riches en eau.
- **Toutes les cultures progressent** sur la période (en moyenne de 6 à 7,7 t/ha toutes cultures confondues). Le **maïs** connaît la plus forte hausse (+75 %).
- Les rendements les plus élevés sont observés pour la **pomme de terre en Belgique, en Nouvelle-Zélande et en Suisse** (près de 50 t/ha), et pour le plantain au Belize.

### 4.2 Le pays compte presque autant que la culture

Pour une même culture, les écarts entre pays sont considérables. Par exemple, le blé produit en médiane **7,7 t/ha au Royaume-Uni** contre **1,6 t/ha au Botswana**, soit près de 5 fois plus. Le pays résume tout ce que les données ne mesurent pas directement : qualité des sols, mécanisation, irrigation, accès aux engrais et savoir-faire.

### 4.3 Le climat et les pesticides : des effets réels mais secondaires

- **Pesticides** : lien positif avec le rendement, net pour les **céréales et le soja** (maïs, soja, sorgho, riz, pomme de terre), quasi nul pour les tubercules tropicaux. **Attention, c'est une corrélation, pas une preuve d'efficacité** : les pays qui utilisent beaucoup de pesticides sont aussi les plus mécanisés et les plus irrigués.
- **Température** : le rendement baisse quand la température augmente pour la plupart des céréales (maïs, blé, riz, sorgho), alors qu'il est stable ou en hausse pour le manioc et l'igname, des cultures tropicales. Cela reflète surtout l'opposition entre pays tempérés et pays tropicaux.
- **Précipitations** : pas de relation globale. L'effet est faible et change de sens selon la culture.
- **À l'échelle d'un pays** (exemple de la France), ces variables n'expliquent quasiment pas les variations d'une année à l'autre. Elles distinguent les pays entre eux, pas les bonnes et les mauvaises années.

### 4.4 Analyse en composantes principales (ACP)

**[À COMPLÉTER]** : l'ACP demandée dans le cahier des charges n'apparaît pas encore dans les notebooks. Contenu prévu :
- part de variance expliquée par les premières composantes ;
- variables qui contribuent le plus à chaque axe (par exemple : axe 1 = « intensification agricole » porté par les pesticides ; axe 2 = « climat » porté par la température et la pluie) ;
- confrontation avec les importances du modèle (section 5.2) pour valider les variables clés.

---

## 5. Le modèle et sa fiabilité

### 5.1 Performances

Le modèle retenu est une **forêt aléatoire** (*Random Forest*), un ensemble de 200 arbres de décision, choisie après comparaison de plusieurs familles de modèles. Il a été entraîné sur 1990 à 2008, puis évalué **une seule fois** sur 2009 à 2013, des années jamais vues, pour simuler une utilisation réelle.

| Indicateur (test 2009-2013) | Règle simple* | Modèle | Lecture métier |
|---|---|---|---|
| Part des écarts expliquée (R²) | 0,836 | **0,852** | Le modèle explique 85 % des différences de rendement |
| Erreur moyenne (MAE) | 1,65 t/ha | **1,58 t/ha** | Écart typique entre prévision et réalité |
| Prévisions à ±15 % du réel | 47,9 % | **48,5 %** | Une prévision sur deux est assez précise pour décider |

\* *Règle simple : on prédit le rendement médian historique du couple pays × culture.*

**Lecture honnête** : savoir *quelle culture* est plantée *dans quel pays* explique déjà l'essentiel du rendement. Le modèle n'apporte qu'un **gain modeste** par rapport à cette règle simple (−4 % d'erreur). Il **sous-estime légèrement** les rendements récents (d'environ 1 t/ha en moyenne), car il ne tient pas compte de la progression des rendements dans le temps.

**[À COMPLÉTER]** : erreur par culture (calculée dans le cycle 5, sorties non sauvegardées) pour indiquer les cultures sur lesquelles l'outil est le plus et le moins fiable.

### 5.2 Variables clés

| Variable | Importance dans le modèle | Levier pour l'agriculteur ? |
|---|---|---|
| Culture | **55 %** | **Oui** : c'est le choix principal |
| Pays | 20 % | Non (contexte subi) |
| Climat (pluie, température, aridité) | 13 % | Non, mais on peut l'anticiper |
| Pesticides | 11 % | Oui, partiellement |
| Indicateurs de données manquantes | 1 % | — |

*L'importance indique à quel point le modèle s'appuie sur une variable, pas dans quel sens elle joue ni si elle en est la cause.*

**[À COMPLÉTER]** : importances recalculées sur le modèle final du cycle 5.

---

## 6. Recommandations

### 6.1 Pour les agriculteurs et les conseillers

1. **Le choix de la culture est le premier levier de rendement.** Utiliser la fonction *Recommandation* pour comparer les cultures dans son contexte avant de décider.
2. **Lire le classement en relatif, pas seulement en tonnes.** Une pomme de terre affichera toujours plus de t/ha qu'un blé. Pour savoir si un contexte est *favorable* à une culture, comparer son rendement prédit à son rendement habituel (indice relatif : au-dessus de 1, le contexte est plus favorable que la moyenne mondiale pour cette culture).
3. **Privilégier les cultures déjà pratiquées dans le pays.** Le modèle peut recommander une culture jamais cultivée localement (par exemple, l'igname ou le manioc en France) : c'est une extrapolation, sans réalité agronomique.
4. **Considérer les pesticides avec prudence.** Les données montrent un lien positif pour les céréales et le soja, mais elles ne prouvent pas qu'ajouter des pesticides augmente le rendement. Leur coût, leur impact environnemental et la réglementation doivent entrer dans la décision.
5. **Utiliser la prévision comme un ordre de grandeur.** L'erreur typique est de 1,6 t/ha ; une prévision sur deux s'écarte de plus de 15 % du réel.

### 6.2 Pour passer du rendement au profit

Le rendement seul ne dit pas quelle culture **rapporte le plus**. Pour une vraie recommandation de rentabilité :

1. **Ajouter les prix de vente par culture et par pays** (par exemple, les prix à la production de la FAO) et calculer un **chiffre d'affaires estimé par hectare** = rendement prédit × prix.
2. **Ajouter les coûts de production** (semences, intrants, main-d'œuvre) pour estimer une **marge par hectare**.
3. **Classer les cultures par marge**, en affichant aussi le niveau d'incertitude de la prévision.

**[À COMPLÉTER]** : exemple chiffré sur un pays avec des prix indicatifs.

### 6.3 Pour améliorer l'outil (feuille de route)

| Priorité | Évolution | Bénéfice attendu |
|---|---|---|
| Haute | Prendre en compte l'**année** dans le modèle | Erreur réduite de **16 %** par rapport à la règle simple, R² de 0,85 à **0,88** (déjà mesuré) |
| Haute | Intégrer **prix et coûts** | Passer d'un classement en rendement à un classement en **rentabilité** |
| Moyenne | Filtrer ou signaler les **cultures absentes du pays** | Éviter les recommandations irréalistes |
| Moyenne | Afficher l'**indice relatif** dans la recommandation | Comparaison équitable entre tubercules et céréales |
| Moyenne | Alléger le modèle (720 Mo aujourd'hui) | Déploiement plus simple et moins coûteux |
| Long terme | Ajouter des données **à la parcelle** (sol, irrigation, engrais, météo locale) | Des conseils propres à chaque exploitation, et non plus à chaque pays |

---

## 7. Limites à communiquer aux clients

- **Échelle** : les données sont nationales. L'outil estime le rendement *typique d'un pays* dans des conditions données, pas celui d'une parcelle précise.
- **Période** : les données s'arrêtent en 2013. Les rendements ayant progressé depuis, les prévisions sont probablement **en dessous** des niveaux actuels.
- **Corrélation n'est pas causalité** : l'outil décrit des tendances observées, il ne garantit pas l'effet d'une action (augmenter les pesticides, par exemple).
- **Variables absentes** : engrais, irrigation, sols, variétés et prix ne sont pas pris en compte.

---

## Annexe : glossaire

| Terme | Signification |
|---|---|
| hg/ha | Hectogrammes par hectare ; 10 000 hg/ha = 1 t/ha |
| R² | Part des différences de rendement expliquée par le modèle (1 = parfait, 0 = aucune explication) |
| MAE | Erreur moyenne, en valeur absolue, entre prévision et réalité |
| RMSE | Erreur quadratique moyenne : comme la MAE, mais pénalise davantage les grosses erreurs |
| Forêt aléatoire | Modèle qui combine les prévisions de nombreux arbres de décision |
| Indice relatif | Rendement prédit ÷ rendement médian mondial de la culture |
