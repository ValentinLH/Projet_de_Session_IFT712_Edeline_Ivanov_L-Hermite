from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from sklearn.calibration import LabelEncoder

# Créez une classe abstraite pour la stratégie
class StrategieClassification(ABC):   
    @abstractmethod
    def entrainer(self, x_train, t_train):
        pass

    @abstractmethod
    def prediction(self, x):
        pass

    @abstractmethod
    def parametres(self):
        pass

    @abstractmethod
    def erreur(self,t, prediction):
        pass
    
    @abstractmethod
    def afficher(self, x_train, t_train, x_test, t_test):
        pass
    

# Classe ClassifieurLineaire avec les méthodes nécessaires pour travailler avec des stratégies de classification
class ClassifieurLineaire:
    def __init__(self, strategie : StrategieClassification):
        """
        Algorithmes de classification lineaire

        La classe prend  une instance de la stratégie de classification.
        """
        self.strategie = strategie

    def entrainement(self, x_train, t_train):
        # Utilisez la stratégie pour l'entraînement
        self.strategie.entrainer(x_train, t_train)

    def prediction(self, x):
        # Utilisez la stratégie pour la prédiction
        if not isinstance(x, (list, np.ndarray)):
            x = [x] 
        
        return self.strategie.prediction(x)

    def erreur(self, t, prediction):
        # Utilisez la stratégie pour calculer l'erreur
        return self.strategie.erreur(t, prediction)

    def afficher_donnees_et_modele(self, x_train, t_train, x_test, t_test):
        """
        afficher les donnees et le modele

        x_train, t_train : donnees d'entrainement
        x_test, t_test : donnees de test
        """
        #self.strategie.afficher(x_train, t_train, x_test, t_test)
        self.afficher(x_train, t_train, x_test, t_test)

    def parametres(self):
        """
        Retourne les paramètres du modèle
        """
        return self.strategie.parametres()
    
    def get_hyperparametres(self):
        """
        Retourne les hyperparamètres du modèle
        """
        return self.strategie.get_hyperparametres()
    
    def set_hyperparametres(self, hyperparametres_list):
        """
        definit les hyperparamètres du modèle
        """
        self.strategie.set_hyperparametres(hyperparametres_list)

    def afficher(self, x_train, t_train, x_test, t_test):
        le = LabelEncoder()
        t_train_encoded = le.fit_transform(t_train)
        t_test_encoded = le.transform(t_test)

        h = 0.05
        x_min, x_max = x_train[:, 0].min() - .5, x_train[:, 0].max() + .5
        y_min, y_max = x_train[:, 1].min() - .5, x_train[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Utiliser LinearNDInterpolator pour interpoler les données
        points = np.column_stack((x_train[:, 0], x_train[:, 1]))
        values = x_train[:,2:]
        
        interpolator = LinearNDInterpolator(points, values)
        grid_xy = np.c_[xx.ravel(), yy.ravel()]
        grid_dim = interpolator(grid_xy)
        grid_tot = np.c_[grid_xy,grid_dim]
        grid_tot[np.isnan(grid_tot)] = 0
        grid_z = self.prediction(grid_tot)
        
        Z = le.transform(grid_z)
        # Remettre les résultats en forme pour le tracé
        Z = Z.reshape(xx.shape)

        plt.figure(0)

        plt.figure(figsize=(14, 8))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
        plt.scatter(x_train[:, 0], x_train[:, 1], c=t_train_encoded, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())

        plt.title('Frontières de décision - Ensemble d\'Entrainement')
        
        h = 0.05
        x_min, x_max = x_test[:, 0].min() - .5, x_test[:, 0].max() + .5
        y_min, y_max = x_test[:, 1].min() - .5, x_test[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Utiliser LinearNDInterpolator pour interpoler les données
        points = np.column_stack((x_test[:, 0], x_test[:, 1]))
        values = x_test[:,2:]
        
        interpolator = LinearNDInterpolator(points, values)
        grid_xy = np.c_[xx.ravel(), yy.ravel()]
        grid_dim = interpolator(grid_xy)
        grid_tot = np.c_[grid_xy,grid_dim]
        grid_tot[np.isnan(grid_tot)] = 0
        grid_z = self.prediction(grid_tot)
        
        Z = le.transform(grid_z)
        # Remettre les résultats en forme pour le tracé
        Z = Z.reshape(xx.shape)

        plt.figure(1)

        plt.figure(figsize=(14, 8))
        plt.close(0)
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
        plt.scatter(x_test[:, 0], x_test[:, 1], c=t_test_encoded, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())

        plt.title('Frontières de décision - Données de test')
        plt.show()    