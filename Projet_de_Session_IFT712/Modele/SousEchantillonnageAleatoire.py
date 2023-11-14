from .RechercheHyperparameter import StrategyRechercheHyperparameter
from sklearn.model_selection import train_test_split
import numpy as np

class SousEchantillonnage(StrategyRechercheHyperparameter):
    def __init__(self, proportion_Test, proportion_Validation):
        """
        Strategie de recherche d'hyperparametre utilisant le sous echantillonnage aleatoire

        :param proportion_Test: proportion des données de test en fonction des données d'entrée, doit etre compris entre 0 et 1
        :param proportion_Valid: proportion des données de Validation en fonction des données d'entrées, doit etre compris entre 0 et 1
        """
        self.proportion_Test = proportion_Test
        self.proportion_Validation = proportion_Validation

    def recherche(self, X, T):
        """
        Decompose les données en base d'entrainement, validation et test en utilisant le sous echantillonnage aleatoire

        :param X: Données d'entrees
        :param T: Etiquette des donnees d'entrees
        """

        # Melange aleatoire des donnees
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        T = T[indices]
         
        # On utilise la fonction train_test_split de sklearn pour separer les donnees en donnees d'entrainement, de validation et de test
        X_Entrainement, X_Temporaire, T_Entrainement, T_temporaire = train_test_split(X, T, test_size=self.proportion_Test)

        X_Test, X_Validation, T_Test, T_Validation = train_test_split(X_Temporaire, T_temporaire, 
                                                                      test_size=(self.proportion_Validation/(1-self.proportion_Test)))
                                                                              
        return X_Entrainement, X_Validation, X_Test, T_Entrainement, T_Validation, T_Test