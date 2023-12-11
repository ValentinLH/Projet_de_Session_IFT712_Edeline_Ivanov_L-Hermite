import torch
from .RechercheHyperparameter import StrategyRechercheHyperparameter
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from itertools import product
import numpy as np


class BootstrapValidation(StrategyRechercheHyperparameter):
    def __init__(self, n_bootstrap, k_fold):
        """
        Strategie de recherche d'hyperparamètres utilisant le bootstrap et la validation croisée.

        :param n_bootstrap: Nombre d'échantillons bootstrap à créer.
        :param k_fold: Nombre de folds pour la validation croisée.
        """
        self.n_bootstrap = n_bootstrap
        self.k_fold = k_fold

    def recherche(self, modele, X, T):
        """
        Réalise une recherche des hyperparamètres utilisant le bootstrap et la validation croisée.

        :param modele: Modèle dont on souhaite rechercher les hyperparamètres.
        :param X: Données d'entrée.
        :param T: Étiquettes des données d'entrée.
        """

        # Récupération des hyperparamètres du modèle
        hyperparametres = modele.get_hyperparametres()

        nbr_iterations = np.prod([len(i) for i in hyperparametres])

        print("########################## Début de la recherche - BootstrapValidation ##########################")
        print("Il y aura ", nbr_iterations, " iterations")

        # Création d'une instance contenant toutes les suites d'hyperparamètres possibles
        hyperparameters_combinaisons = product(*hyperparametres)

        first_line = next(hyperparameters_combinaisons)

        meilleur_precision = 0.0
        meilleur_hyperparametres = first_line

        compteur = 0

        for parametres in (first_line, *hyperparameters_combinaisons):
            precision_total = 0.0
            compteur += 1

            if (compteur % 10 == 0):
                print("iterations ", compteur, " sur ", nbr_iterations)

            for _ in range(self.n_bootstrap):
                # Utilisation du bootstrap pour créer un nouvel ensemble d'entraînement
                X_bootstrap, T_bootstrap = resample(X, T, random_state=42)

                # Utilisation de KFold pour la validation croisée
                kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=42)

                for entrainement_index, validation_index in kf.split(X_bootstrap):
                    X_Entrainement, X_Validation = X_bootstrap[entrainement_index], X_bootstrap[validation_index]
                    T_Entrainement, T_Validation = T_bootstrap[entrainement_index], T_bootstrap[validation_index]

                    modele.set_hyperparametres(parametres)
                    modele.entrainement(X_Entrainement, T_Entrainement)

                    predictions = modele.prediction(X_Validation)

                    if isinstance(T_Validation, torch.Tensor):
                        # Transformation du one hot vector en valeur de classe pour le calcul d'accuracy
                        _, t_valid_pred = torch.max(T_Validation, 1)
                        T_Validation = t_valid_pred.tolist()

                    precision_total += accuracy_score(T_Validation, predictions)

            precision_moyenne = precision_total / (self.n_bootstrap * self.k_fold)

            if precision_moyenne > meilleur_precision:
                meilleur_precision = precision_moyenne
                meilleur_hyperparametres = parametres
                print("iterations ", compteur, " sur ", nbr_iterations)
                print("precision amélioré: ", meilleur_precision, "\tavec ces paramètres: ", meilleur_hyperparametres)

        print("########################## Fin de la recherche ##########################")
        print("meilleur hyperparametres: ", meilleur_hyperparametres)
        print("precisin trouvée: ", meilleur_precision)
        modele.set_hyperparametres(meilleur_hyperparametres)
