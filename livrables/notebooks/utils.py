import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate(y_true, y_pred):
    score_r2 = r2_score(y_true, y_pred)
    score_mae = mean_absolute_error(y_true, y_pred)
    score_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        'r2': score_r2,
        'mae': score_mae,
        'rmse': score_rmse,
    }

def test_train_time_split(dataset):
    # 1. Trier les années par ordre croissant et compter le nombre de lignes par année
    counts_par_annee = dataset['Year'].value_counts().sort_index()

    # 2. Calculer le seuil de 80% du nombre total de lignes
    total_lignes = len(dataset)
    seuil_80pct = total_lignes * 0.80

    # 3. Première année où le cumul dépasse 80% -> début du test
    cumul_lignes = counts_par_annee.cumsum()
    split_year = cumul_lignes[cumul_lignes >= seuil_80pct].index[0]

    # 4. Séparation stricte passé / futur
    X = dataset.drop(columns=['hg/ha'])
    y = dataset['hg/ha']

    train_mask = X['Year'] < split_year

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    print(f"Année de césure calculée : {split_year}")
    print(f"Proportion réelle train : {train_mask.mean():.2%} ({len(X_train)} lignes)")
    print(f"Proportion réelle test  : {(~train_mask).mean():.2%} ({len(X_test)} lignes)")

    return X_train, X_test, y_train, y_test