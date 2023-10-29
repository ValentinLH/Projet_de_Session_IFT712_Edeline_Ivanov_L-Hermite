from sklearn.linear_model import Perceptron as SKlearnPerceptron
from .ClassifieurLineaire import StrategieClassification

class Perceptron(StrategieClassification):
    def __init__(self, learning_rate=0.01, max_iterations=1000,penalty='l2'):
        """
        Stratégie de classification utilisant le Perceptron de scikit-learn.

        :param learning_rate: Taux d'apprentissage pour le Perceptron.
        :param max_iterations: Nombre maximal d'itérations pour l'entraînement.
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.penalty=penalty
        self.perceptron_model = None

    def entrainer(self, x_train, t_train):
        """
        Entraîne le modèle de classification Perceptron de scikit-learn.

        :param x_train: Les données d'entraînement.
        :param t_train: Les étiquettes de classe cibles.
        """
        self.perceptron_model = SKlearnPerceptron(eta0=self.learning_rate, max_iter=self.max_iterations,penalty=self.penalty)
        self.perceptron_model.fit(x_train, t_train)
        self.w = self.perceptron_model.coef_[0]
        self.w_0 = self.perceptron_model.intercept_[0]

    def prediction(self, x):
        """
        Prédit la classe d'une nouvelle donnée x.

        :param classifieur: Une instance de ClassifieurLineaire.
        :param x: La donnée d'entrée à classifier.
        :return: 1 si la classe prédite est positive, -1 sinon.
        """
        if self.perceptron_model is not None:
            return self.perceptron_model.predict([x])[0]
        return 0  # Valeur par défaut si le modèle n'est pas encore entraîné

    def erreur(self, t, prediction):
        """
        Calcule l'erreur de classification.

        :param t: L'étiquette de classe réelle.
        :param prediction: La prédiction du modèle.
        :return: 1 si l'erreur est commise, 0 sinon.
        """
        return 1 if t != prediction else 0
