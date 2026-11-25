import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AgritechFeatureEngineering(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        X_out['aridity'] = (X_out['avg_temp'] + 10) / (X_out['average_rain_fall_mm_per_year'] + 1)
        X_out['avg_temp_square'] = np.square(X_out['avg_temp'])

        return X_out