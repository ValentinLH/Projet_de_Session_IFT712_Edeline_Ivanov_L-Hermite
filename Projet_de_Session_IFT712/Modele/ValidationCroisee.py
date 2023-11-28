from sklearn.model_selection import KFold, train_test_split
from .RechercheHyperparameter import StrategyRechercheHyperparameter

class ValidationCroisee(StrategyRechercheHyperparameter):
    def __init__(self, k, proportion_Test, proportion_Validation):
        """
        Strategie de recherche d'hyperparametre utilisant la validation croisee

        :param k: parametre k de la validation croisee
        :param proportion_Test: proportion des données de test en fonction des données d'entrée, doit etre compris entre 0 et 1
        :param proportion_Valid: proportion des données de Validation en fonction des données d'entrées, doit etre compris entre 0 et 1
        """
        self.k = k
        self.proportion_Test = proportion_Test
        self.proportion_Validation = proportion_Validation

    def recherche(self, X, T):
        """
        Decompose les données en base d'entrainement, validation et test en utilisant la validation croisee

        :param X: Données d'entrees
        :param T: Etiquette des donnees d'entrees
        """

        validation_Croisee = KFold(n_splits=self.k, shuffle=True)

        X_Entrainement, X_Validation, X_Test, T_Entrainement, T_Validation, T_Test = ([] for i in range(6))

        for index_Entrainement, index_Validation_Test in validation_Croisee.split(X):
            X_Entrainement_Temp = X[index_Entrainement]
            X_Valid_Test_Temp = X[index_Validation_Test]
            T_Entrainement_Temp = T[index_Entrainement]
            T_Valid_Test_Temp = T[index_Validation_Test]

            X_Valid_Temp, X_Test_Temp, T_Valid_Temp, T_Test_Temp = train_test_split(
                X_Valid_Test_Temp, T_Valid_Test_Temp, test_size=(self.proportion_Validation/(1-self.proportion_Test)))
            
            X_Entrainement.append(X_Entrainement_Temp)
            X_Validation.append(X_Valid_Temp)
            X_Test.append(X_Test_Temp)
            T_Entrainement.append(T_Entrainement_Temp)
            T_Validation.append(T_Valid_Temp)
            T_Test.append(T_Test_Temp)
        
        return X_Entrainement, X_Validation, X_Test, T_Entrainement, T_Validation, T_Test