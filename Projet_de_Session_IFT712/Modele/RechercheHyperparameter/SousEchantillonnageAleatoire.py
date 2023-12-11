import torch
from .RechercheHyperparameter import StrategyRechercheHyperparameter
from sklearn.model_selection import train_test_split
import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score


class SousEchantillonnage(StrategyRechercheHyperparameter):
    def __init__(self, k, proportion_validation):
        """
        Strategie de recherche d'hyperparametre utilisant le sous echantillonnage aleatoire

        :param k: nombre d'itérations à réaliser par ensemble de paramètres
        :param proportion_Valid: proportion des données de Validation en fonction des données d'entrées
               doit etre compris entre 0 et 1
        """
        self.k = k
        self.proportion_validation = proportion_validation

    def recherche(self, modele, X, T):
        """
        Réalise une recherche des hyperparamètres utilisant le sous-echantillonnage aléatoire

        :param modele: modèle dont on souhaite rechercher les hyperparamètres
        :param X: Données d'entrées
        :param T: Etiquette des donnees d'entrées
        """

        # Récupération des hyperparamètres du modèle
        hyperparametres = modele.get_hyperparametres()

        nbr_iterations = np.prod([len(i) for i in hyperparametres])
        compteur = 1

        print(
            "########################## "
            "Début de la recherche - Sous Echantillonnage Aleatoire "
            "##########################")

        print("Il y aura ", nbr_iterations, " iterations")

        # Création d'une instance contenant toutes les suites d'hyperparamètres possibles
        hyperparameters_combinaisons = product(*hyperparametres)

        premiere_ligne = next(hyperparameters_combinaisons)

        meilleur_precision = 0.0
        meilleur_hyperparametres = premiere_ligne

        for parametres in (premiere_ligne, *hyperparameters_combinaisons):

            precision_total = 0.0

            compteur += 1

            if (compteur % 10 == 0):
                print("iterations ", compteur, " sur ", nbr_iterations)

            for j in range(self.k):

                # Mélange aléatoire des données
                indices = np.arange(len(X))
                np.random.shuffle(indices)
                X = X[indices]
                T = T[indices]

                X_Entrainement, X_Validation, T_Entrainement, T_Validation = train_test_split(X, T,
                                                                                              test_size=self.proportion_validation)

                modele.set_hyperparametres(parametres)
                modele.entrainement(X_Entrainement, T_Entrainement)

                predictions = modele.prediction(X_Validation)

                if isinstance(T_Validation,torch.Tensor):
                    # Transformation du one hot vector en valeur de classe pour le calcul d'accuracy
                    _, t_valid_pred = torch.max(T_Validation, 1)
                    T_Validation = t_valid_pred.tolist()

                precision_total += accuracy_score(T_Validation, predictions)

            precision_moyenne = precision_total / self.k

            if (precision_moyenne > meilleur_precision):
                meilleur_precision = precision_moyenne
                meilleur_hyperparametres = parametres
                print("iterations ", compteur, " sur ", nbr_iterations)
                print("precision amélioré: ", meilleur_precision, "\tavec ces paramètres: ", meilleur_hyperparametres)

        print("########################## Fin de la recherche ##########################")
        print("meilleur hyperparametres: ", meilleur_hyperparametres)
        print("precisin trouvée: ", meilleur_precision)
        modele.set_hyperparametres(meilleur_hyperparametres)
