from sklearn.model_selection import KFold, train_test_split
from .RechercheHyperparameter import StrategyRechercheHyperparameter
import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

class ValidationCroisee(StrategyRechercheHyperparameter):
    def __init__(self, k):
        """
        Strategie de recherche d'hyperparametre utilisant la validation croisee

        :param k: parametre k de la validation croisee
        """
        self.k = k

    def recherche(self, modele, X, T):
        """
        Réalise une recherche des hyperparametres utilisant la validation croisée

        :param modele: modèle dont on souhaite rechercher les hyperparamètres
        :param X: Données d'entrees
        :param T: Etiquette des donnees d'entrees
        """

        #Récupération des hyperparamètres du modèle
        hyperparametres = modele.get_hyperparametres()

        #Création d'une instance contenant toutes les suites d'hyperparamètres possibles
        hyperparameters_combinaisons = product(*hyperparametres)

        first_line = next(hyperparameters_combinaisons)

        meilleur_precision = 0.0
        meilleur_hyperparametres = first_line

        for parametres in (first_line, *hyperparameters_combinaisons):

            precision_total = 0.0

            # Utilisation de KFold pour la validation croisée
            kf = KFold(n_splits=self.k, shuffle=True, random_state=42)

            for entrainement_index, validation_index in kf.split(X):

                X_Entrainement, X_Validation = X[entrainement_index], X[validation_index]
                T_Entrainement, T_Validation = T[entrainement_index], T[validation_index]

                modele.set_hyperparametres(parametres)
                modele.entrainement(X_Entrainement, T_Entrainement)

                predictions = modele.prediction(X_Validation)
                precision_total += accuracy_score(T_Validation, predictions)

            precision_moyenne = precision_total / self.k

            if precision_moyenne > meilleur_precision:
                meilleur_precision = precision_moyenne
                meilleur_hyperparametres = parametres

        print("meilleur hyperparametres: ", meilleur_hyperparametres)
        modele.set_hyperparametres(meilleur_hyperparametres)
