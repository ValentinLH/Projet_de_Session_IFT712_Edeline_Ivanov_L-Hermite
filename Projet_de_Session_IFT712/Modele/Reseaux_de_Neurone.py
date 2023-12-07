import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Projet_de_Session_IFT712.Modele.ClassifieurLineaire import StrategieClassification
from Projet_de_Session_IFT712.Modele.data import TrainData


class Reseaux_Neurones(StrategieClassification) :

    def __init__(self,hidden_layer_size = (48,12,3),solver = "adam", activation = "relu" ,alpha = 0.0001,learning_rate_init=0.001,max_iter=200,beta_1=0.9, beta_2=0.999, epsilon=1e-08, momentum = 0.9, power_t = 0.5):

        self.hidden_layer_size = hidden_layer_size
        self.solver = solver
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.momentum = momentum
        self.power_t = power_t

        self.MLP_model = None

        self.W = None
        self.W0 = None

    def entrainer(self, x_train, t_train):

        self.MLP_model = MLPClassifier(hidden_layer_sizes= self.hidden_layer_size, solver=self.solver , activation=self.activation, alpha=self.alpha, learning_rate_init=self.learning_rate_init, max_iter=self.max_iter, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon,momentum = self.momentum, power_t = self.power_t)
        self.MLP_model = self.MLP_model.fit(x_train,t_train)
        self.W0 = self.MLP_model.intercepts_
        self.W = self.MLP_model.coefs_

    def prediction(self, x):
        return self.MLP_model.predict([x])

    def erreur(self,t, prediction):
        return 1 if t != prediction else 0

    def parametres(self):
        return self.W0, self.W
    def afficher(self, x_train, t_train, x_test, t_test):
        return
