from sklearn.neural_network import MLPClassifier
from Modele.ClassifieurLineaire import StrategieClassification
import numpy as np


class Reseaux_Neurones(StrategieClassification):

    def __init__(self, hidden_layer_size=(48, 12, 3), solver="adam", activation="relu", alpha=0.0001,
                 learning_rate="constant", learning_rate_init=0.001, max_iter=500, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-08, momentum=0.9, power_t=0.5, max_fun=15000):
        """
        :param hidden_layer_size: structure du reseau de neurone
        :param solver: le type de solveur utilise pour que le modele apprend
        :param activation: la fontion d'activation
        :param alpha: le taux de regularisartion utilise par le reseaux de neurone
        :param learning_rate: le type du pas d'apprentissage
        :param learning_rate_init: la valeur initiale du pas d'apprentissage
        :param max_iter: le nombre d'iterations maximum que peut realiser le modele
        :param beta_1: la valeur du parametre beta1 pour Adam
        :param beta_2: la valeur du parametre beta2 pour Adam
        :param epsilon: la valeur du parametre epsilon pour Adam
        :param momentum: la valeur du momentum pour la descente de gradient stochastique
        :param power_t : la valeur l'exposent pour le "inverse scaling learning rate"
        :param max_fun: la valeur maximum d'appel que peut faire le solveur lbfgs
        """
        # Parametre generaux du MLP
        self.hidden_layer_size = hidden_layer_size
        self.solver = solver
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        # Parametre pour le solveur Adam
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # Parametre pour le solveur classique de la descente de gradient
        self.momentum = momentum

        # Parametre pour quand le learning_rate utilise est invscaling
        self.power_t = power_t

        # Parametre pour le solveur lbfgs
        self.max_fun = max_fun

        self.MLP_model = None

        self.W = None
        self.W0 = None

    def entrainer(self, x_train, t_train):
        """
        Entraine le modele de classification Multi layer Perceptron de scikit-learn.

        :param x_train: Les donnees d'entraînement.
        :param t_train: Les étiquettes de classe cibles.
        """

        self.MLP_model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_size, solver=self.solver,
                                       activation=self.activation, alpha=self.alpha, learning_rate=self.learning_rate,
                                       learning_rate_init=self.learning_rate_init, max_iter=self.max_iter,
                                       beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon,
                                       momentum=self.momentum, power_t=self.power_t, max_fun=self.max_fun)
        self.MLP_model = self.MLP_model.fit(x_train, t_train)
        self.W0 = self.MLP_model.intercepts_
        self.W = self.MLP_model.coefs_

    def prediction(self, X):
        """
        Predit la classe de nouvelle donnees X.

        :param X: Les donnees d'entree à classifier.
        :return: retourne les classes associes aux données
        """
        predictions = [self.MLP_model.predict([x]) for x in X]
        return predictions

    def erreur(self, t, prediction):
        """
           Calcule l'erreur de classification.

           :param t: etiquette de classe.
           :param prediction: La prediction du modele.
           :return: 1 si l'erreur est commise, 0 sinon.
       """

        return 1 if t != prediction else 0

    def parametres(self):
        """
            Retourne les parametres du classifieur

            :return: dictionnaire composé des parametres associé à leur valeur
        """
        return self.W0, self.W

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
        # parametre generaux du MLPClassifier
        learning_rate_type_liste = ["constant", "invscaling", "adaptive"]
        learning_rate_liste = np.linspace(0.001, 1, 5)
        max_iterations_liste = np.linspace(200, 1000, 5).astype(int)
        fonction_activation = ["tanh", "relu"]
        solveur_liste = ["lbfgs", "adam", "sgd"]

        return [learning_rate_type_liste, learning_rate_liste, max_iterations_liste, fonction_activation, solveur_liste]

    def set_hyperparametres(self, hyperparametres_list):
        """
        Met à jour les valeurs des hyperparamètres

        :param hyperparametres_list: liste contenant les nouvelles valeurs des hyperparamètres
        """
        self.learning_rate = hyperparametres_list[0]
        self.learning_rate_init = hyperparametres_list[1]
        self.max_iter = hyperparametres_list[2]
        self.activation = hyperparametres_list[3]
        self.solver = hyperparametres_list[4]
