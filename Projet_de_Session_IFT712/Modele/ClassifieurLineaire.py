import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# Créez une classe abstraite pour la stratégie
class StrategieClassification:
    
    @abstractmethod
    def entrainer(self, x_train, t_train):
        pass


    @abstractmethod
    def prediction(self, x):
        pass


    @abstractmethod
    def parametres(self):
        pass

    @staticmethod
    def erreur(t, prediction):
        pass
    

# Classe ClassifieurLineaire avec les méthodes nécessaires pour travailler avec des stratégies de classification
class ClassifieurLineaire:
    def __init__(self, strategie):
        """
        Algorithmes de classification lineaire

        La classe prend  une instance de la stratégie de classification.
        """
        self.w = None
        self.w_0 = None
        self.strategie = strategie

    def entrainement(self, x_train, t_train):
        # Utilisez la stratégie pour l'entraînement
        self.strategie.entrainer(self, x_train, t_train)

    def prediction(self, x):
        # Utilisez la stratégie pour la prédiction
        return self.strategie.prediction(self, x)

    @staticmethod
    def erreur(t, prediction):
        # Utilisez la stratégie pour calculer l'erreur
        return StrategieClassification.erreur(t, prediction)

    def afficher_donnees_et_modele(self, x_train, t_train, x_test, t_test):
        """
        afficher les donnees et le modele

        x_train, t_train : donnees d'entrainement
        x_test, t_test : donnees de test
        """
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=t_train * 100 + 20, c=t_train)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Training data')

        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=t_test * 100 + 20, c=t_test)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Testing data')

        plt.show()

    def parametres(self):
        """
        Retourne les paramètres du modèle
        """
        return self.w_0, self.w
