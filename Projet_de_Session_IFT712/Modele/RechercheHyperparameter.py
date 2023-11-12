from abc import ABC, abstractmethod

class StrategyRechercheHyperparameter(ABC):
    @abstractmethod
    def recherche(self, model, X, T):
        pass

class RechercheHyperparameter:
    def __init__(self, strategie : StrategyRechercheHyperparameter):
        self.strategie = strategie

    def recherche(self, model, X, T):
        self.strategie.recherche(model, X, T)