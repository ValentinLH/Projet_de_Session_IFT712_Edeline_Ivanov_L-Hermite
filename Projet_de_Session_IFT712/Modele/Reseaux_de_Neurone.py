import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Projet_de_Session_IFT712.Modele.ClassifieurLineaire import StrategieClassification
from Projet_de_Session_IFT712.Modele.data import TrainData
import numpy as np

class Reseaux_Neurones(StrategieClassification) :

    def __init__(self,hidden_layer_size = (48,12,3),solver = "adam", activation = "relu" ,alpha = 0.0001,learning_rate = "constant",learning_rate_init=0.001,max_iter=500,beta_1=0.9, beta_2=0.999, epsilon=1e-08, momentum = 0.9, power_t = 0.5, max_fun = 15000):

        #Parametre generaux du MLP
        self.hidden_layer_size = hidden_layer_size
        self.solver = solver
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        #Parametre pour le solveur Adam
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        #Parametre pour le solveur classique de la descente de gradient
        self.momentum = momentum

        #Parametre pour quand le learning_rate utilise est invscaling
        self.power_t = power_t

        #Parametre pour le solveur lbfgs
        self.max_fun = max_fun

        self.MLP_model = None

        self.W = None
        self.W0 = None

    def entrainer(self, x_train, t_train):

        self.MLP_model = MLPClassifier(hidden_layer_sizes= self.hidden_layer_size, solver=self.solver , activation=self.activation, alpha=self.alpha,learning_rate = self.learning_rate ,learning_rate_init=self.learning_rate_init, max_iter=self.max_iter, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon,momentum = self.momentum, power_t = self.power_t,max_fun=self.max_fun)
        self.MLP_model = self.MLP_model.fit(x_train,t_train)
        self.W0 = self.MLP_model.intercepts_
        self.W = self.MLP_model.coefs_

    def prediction(self, X):
        predictions = [self.MLP_model.predict([x]) for x in X]
        return predictions

    def erreur(self, t, prediction):
        return 1 if t != prediction else 0

    def parametres(self):
        return self.W0, self.W
    def afficher(self, x_train, t_train, x_test, t_test):
        #Pas besoin de la coder car il y a un affichage generique
        return

    def get_hyperparametres(self):

        #parametre generaux du MLPClassifier
        learning_rate_type_liste = ["constant", "invscaling", "adaptive"]
        learning_rate_liste = np.linspace(0.001, 1, 5)  # np.array([0.01]) #np.logspace(-4, 0, 5) #np.linspace(0.001, 0.01, 10)  # np.array([0.01])
        max_iterations_liste = np.linspace(200, 1000, 5).astype(int)
        fonction_activation = ["tanh", "relu"]
        solveur_liste = ["lbfgs", "adam","sgd"]

        return [learning_rate_type_liste, learning_rate_liste,max_iterations_liste, fonction_activation,solveur_liste]

    def set_hyperparametres(self, hyperparametres_list):

        self.learning_rate = hyperparametres_list[0]
        self.learning_rate_init = hyperparametres_list[1]
        self.max_iter = hyperparametres_list[2]
        self.activation = hyperparametres_list[3]
        self.solver = hyperparametres_list[4]
