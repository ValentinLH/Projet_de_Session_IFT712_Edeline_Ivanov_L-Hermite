from abc import ABC, abstractmethod


class StrategyRechercheHyperparameter(ABC):
    @abstractmethod
    def recherche(self, model, X, T):
        pass


class RechercheHyperparameter:
    def __init__(self, strategie: StrategyRechercheHyperparameter):
        """
        Algorithmes de recherches d'hyperparamètres

        prends une instance de la stratégie de recherche
        """
        self.strategie = strategie

    def recherche(self, modele, X, T):
        """
        Réalise une recherche d'hyperparamètres
        """
        self.strategie.recherche(modele, X, T)
