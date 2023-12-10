from sklearn.model_selection import KFold
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

        nbr_iterations = np.prod([len(i) for i in hyperparametres])
        compteur = 0

        print("########################## Début de la recherche ##########################")
        print("Il y aura ", nbr_iterations, " iterations")

        #Création d'une instance contenant toutes les suites d'hyperparamètres possibles
        hyperparameters_combinaisons = product(*hyperparametres)

        premiere_ligne = next(hyperparameters_combinaisons)

        meilleur_precision = 0.0
        meilleur_hyperparametres = premiere_ligne

        for parametres in (premiere_ligne, *hyperparameters_combinaisons):

            precision_total = 0.0
            
            compteur += 1

            if (compteur%10 == 0):
                print("iterations ", compteur, " sur ", nbr_iterations)
            
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
                print("iterations ", compteur, " sur ", nbr_iterations)
                print("precision amélioré: ", meilleur_precision, "\tavec ces paramètres: ", meilleur_hyperparametres)

        print("########################## Fin de la recherche ##########################")
        print("meilleur hyperparametres: ", meilleur_hyperparametres)
        print("precisin trouvée: ", meilleur_precision)
        modele.set_hyperparametres(meilleur_hyperparametres)
