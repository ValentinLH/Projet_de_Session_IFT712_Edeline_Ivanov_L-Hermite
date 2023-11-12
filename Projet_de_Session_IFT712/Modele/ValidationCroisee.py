from sklearn.model_selection import cross_val_score
from .RechercheHyperparameter import StrategyRechercheHyperparameter
import numpy as np

class ValidationCroisee(StrategyRechercheHyperparameter):
    def __init__(self, k):
        self.k = k

    def search(self, model, X, T):
        result = cross_val_score(model, X, T, cv=self.k)
        return np.mean(result)
    