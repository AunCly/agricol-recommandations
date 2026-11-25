import pandas as pd

from pipeline import train as pipeline


def scores(rmse):
    return {"r2": 0.8, "mae": 1000.0, "rmse": rmse}


class TestValidation:
    def test_accepte_un_modele_meilleur_que_la_baseline_et_l_actuel(self):
        evaluation = {"modele": scores(30_000), "baseline": scores(32_000)}
        checks = pipeline.validate(evaluation, current_rmse=31_000)
        assert all(checks.values())

    def test_refuse_un_modele_moins_bon_que_la_baseline(self):
        evaluation = {"modele": scores(33_000), "baseline": scores(32_000)}
        checks = pipeline.validate(evaluation, current_rmse=None)
        assert not all(checks.values())

    def test_refuse_un_modele_moins_bon_que_l_actuel(self):
        evaluation = {"modele": scores(31_500), "baseline": scores(40_000)}
        checks = pipeline.validate(evaluation, current_rmse=31_000)
        assert not all(checks.values())

    def test_refuse_un_modele_equivalent_a_l_actuel(self):
        evaluation = {"modele": scores(31_000), "baseline": scores(40_000)}
        checks = pipeline.validate(evaluation, current_rmse=31_000)
        assert not all(checks.values())


class TestEntrainement:
    def test_baseline_utilise_la_mediane_pays_culture(self):
        X_train = pd.DataFrame({"Area": ["France", "France"], "Item": ["Blé", "Blé"]})
        y_train = pd.Series([10, 20])
        X_test = pd.DataFrame({"Area": ["France"], "Item": ["Blé"]})
        assert pipeline.baseline_predictions(X_train, y_train, X_test).tolist() == [15]

    def test_entrainement_et_evaluation_sur_un_echantillon(self):
        # Petit échantillon pour garder le test rapide
        dataset, X_train, X_test, y_train, y_test = pipeline.load_data()
        X_train, y_train = X_train.head(1000), y_train.head(1000)
        X_test, y_test = X_test.head(200), y_test.head(200)

        model = pipeline.train(X_train, y_train)
        evaluation = pipeline.evaluate(model, X_train, y_train, X_test, y_test)

        assert set(evaluation) == {"modele", "baseline"}
        assert evaluation["modele"]["rmse"] > 0
