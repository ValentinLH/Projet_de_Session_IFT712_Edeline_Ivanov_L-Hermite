from sklearn.linear_model import Perceptron as SKlearnPerceptron
from .ClassifieurLineaire import StrategieClassification
import numpy as np


class Perceptron(StrategieClassification):
    def __init__(self, learning_rate=0.01, max_iterations=1000, penalty='l2'):
        """
        Stratégie de classification utilisant le Perceptron de scikit-learn.

        :param learning_rate: Taux d'apprentissage pour le Perceptron.
        :param max_iterations: Nombre maximal d'itérations pour l'entraînement.
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.penalty = penalty
        self.perceptron_model = None

    def entrainer(self, x_train, t_train):
        """
        Entraîne le modèle de classification Perceptron de scikit-learn.

        :param x_train: Les données d'entraînement.
        :param t_train: Les étiquettes de classe cibles.
        """
        self.perceptron_model = SKlearnPerceptron(eta0=self.learning_rate, max_iter=self.max_iterations,
                                                  penalty=self.penalty)
        self.perceptron_model.fit(x_train, t_train)
        self.w = self.perceptron_model.coef_
        self.w_0 = self.perceptron_model.intercept_

    def prediction(self, x):
        """
        Prédit la classe d'une nouvelle donnée x.

        :param x: La donnée d'entrée à classifier.
        :return: 1 si la classe prédite est positive, -1 sinon.
        """
        if self.perceptron_model is not None:
            return self.perceptron_model.predict(x)
        return 0  # Valeur par défaut si le modèle n'est pas encore entraîné

    def erreur(self, t, prediction):
        """
        Calcule l'erreur de classification.

        :param t: L'étiquette de classe réelle.
        :param prediction: La prédiction du modèle.
        :return: 1 si l'erreur est commise, 0 sinon.
        """
        return 1 if t != prediction else 0

    def parametres(self):
        """
        Retourne les paramètres du modèle
        """
        return self.w_0, self.w

    def afficher(self, x_train, t_train, x_test, t_test):
        """
        Methode d'affichage des frontières de décision pour l'ensemble d'entraînement et de test.
        On ne fait plus rien car elle est remplacer par la méthode générique.

        :param x_train: Données d'entraînement.
        :param t_train: Étiquettes d'entraînement.
        :param x_test: Données de test.
        :param t_test: Étiquettes de test.
        """
        pass

    def get_hyperparametres(self):
        """
        Renvoie une liste de valeurs que peuvent prendre les hyperparamètres

        :return: Une liste contenant un ensemble de valeur possible pour chaque hyperparamètres
        """
        learning_rate_liste = np.linspace(0.001, 1, 10)
        max_iterations_liste = np.linspace(500, 1500, 10).astype(int)
        penalty_liste = np.array(['l2'])

        return [learning_rate_liste,
                max_iterations_liste,
                penalty_liste]

    def set_hyperparametres(self, hyperparametres_list):
        """
        Met à jour les valeurs des hyperparamètres

        :param hyperparametres_list: liste contenant les nouvelles valeurs des hyperparamètres
        """
        self.learning_rate = hyperparametres_list[0]
        self.max_iterations = hyperparametres_list[1]
        self.penalty = hyperparametres_list[2]
