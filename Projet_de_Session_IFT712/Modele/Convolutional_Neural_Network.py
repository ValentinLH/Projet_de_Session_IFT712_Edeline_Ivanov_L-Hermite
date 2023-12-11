from Modele.ClassifieurLineaire import StrategieClassification
import numpy as np
from Modele.Net import Net


class Convolutional_Neural_Network(StrategieClassification):
    def __init__(self, lr=0.001, epochs=15, batch_size=64, dropout=0.5):
        """
        Stratégie de classification utilisant un réseau de neurones convolutionnel.

        :param lr: Taux d'apprentissage.
        :param epochs: Nombre d'epochs pour l'entraînement.
        :param batch_size: Taille du batch pour l'entraînement.
        :param dropout: Valeur du Dropout.
        """

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout

        self.CNN = None

    def entrainer(self, x_train, t_train):
        """
        Entraîne le réseau de neurones convolutionnel.

        :param x_train: Les données d'entraînement.
        :param t_train: Les étiquettes de classe cibles.
        """
        self.CNN = Net(lr=self.lr, dropout=self.dropout, epochs=self.epochs, batch_size=self.batch_size)
        self.CNN.entrainer(x_train, t_train)

    def prediction(self, x):
        """
        Prédit la classe d'une nouvelle donnée x.

        :param x: La donnée d'entrée à classifier.
        :return: Prédictions du modèle pour les données d'entrée.
        """
        return self.CNN.prediction(x)

    def parametres(self):
        """
        Retourne les paramètres du modèle.

        Returns:
        - list: Liste des paramètres du modèle.
        """
        return list(self.CNN.parameters())

    def erreur(self, t, prediction):
        """
        Calcule l'erreur de classification.

        :param t: L'étiquette de classe réelle.
        :param prediction: La prédiction du modèle.
        :return: Erreur de classification.
        """
        return self.CNN.erreur(t, prediction)

    def afficher(self, x_train, t_train, x_test, t_test):
        """
        Affiche les résultats du modèle (non implémenté dans cette classe car non nécessaire).

        :param x_train: Données d'entraînement.
        :param t_train: Étiquettes d'entraînement.
        :param x_test: Données de test.
        :param t_test: Étiquettes de test.
        """
        pass

    def get_hyperparametres(self):
        """
        Renvoie une liste de valeurs que peuvent prendre les hyperparamètres.

        Returns:
        - list: Liste contenant un ensemble de valeur possible pour chaque hyperparamètre.
        """
        learning_rate_liste = np.linspace(0.001, 1, 10)
        drop_out_liste = np.linspace(0.2, 0.8, 5).astype(int)
        epoch_liste = np.linspace(1, 15, 10).astype(int)
        taille_du_batch_liste = np.linspace(64, 128, 5).astype(int)

        return [learning_rate_liste, drop_out_liste, epoch_liste, taille_du_batch_liste]

    def set_hyperparametres(self, hyperparametres_list):
        """
        Met à jour les valeurs des hyperparamètres
        :param hyperparametres_list: liste contenant les nouvelles valeurs des hyperparamètres
        """
        self.lr = hyperparametres_list[0]
        self.dropout = hyperparametres_list[1]
        self.epochs = hyperparametres_list[2]
        self.batch_size = int(hyperparametres_list[3])
