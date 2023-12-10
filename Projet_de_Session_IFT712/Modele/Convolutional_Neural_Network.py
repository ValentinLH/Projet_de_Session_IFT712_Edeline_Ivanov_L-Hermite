from Modele.ClassifieurLineaire import StrategieClassification

import torch
import numpy as np
from Projet_de_Session_IFT712.Modele.Net import Net

class Convolutional_Neural_Network(StrategieClassification):
    def __init__(self,lr=0.001, epochs=15, batch_size=64, dropout=0.5):
        """
        :param lr: la valeur du pas d'apprentissage
        :param epochs: le nombre d'epochs realiser lors de l'entrainement
        :param batch_size: la taille du batch utiliser pour l'entrainement
        :param dropout: la valeur du DropOut
        """

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout

        self.CNN = None

    def entrainer(self, x_train, t_train):
        self.CNN = Net(lr=self.lr, dropout=self.dropout, epochs=self.epochs, batch_size=self.batch_size)
        self.CNN.entrainer(x_train, t_train)

    def prediction(self, x):
        return self.CNN.prediction(x)

    def parametres(self):
        return list(self.CNN.parameters())

    def erreur(self, t, prediction):
        return self.CNN.erreur(t, prediction)

    def afficher(self, x_train, t_train, x_test, t_test):
        pass

    def get_hyperparametres(self):
        """
        Renvoie une liste de valeurs que peuvent prendre les hyperparamètres

        :return: Une liste contenant un ensemble de valeur possible pour chaque hyperparamètres
        """

        learning_rate_liste = np.linspace(0.001, 1, 10)
        drop_out_liste = np.linspace(0.2, 0.8, 5).astype(int)
        epoch_liste = np.linspace(1, 15, 10).astype(int)
        taille_du_batch_liste = np.linspace(64, 128, 5).astype(int)

        return [learning_rate_liste, drop_out_liste, epoch_liste,taille_du_batch_liste]


    def set_hyperparametres(self, hyperparametres_list):
        """
        Met à jour les valeurs des hyperparamètres
        :param hyperparametres_list: liste contenant les nouvelles valeurs des hyperparamètres
        """
        self.lr = hyperparametres_list[0]
        self.dropout = hyperparametres_list[1]
        self.epochs = hyperparametres_list[2]
        self.batch_size = hyperparametres_list[3]

