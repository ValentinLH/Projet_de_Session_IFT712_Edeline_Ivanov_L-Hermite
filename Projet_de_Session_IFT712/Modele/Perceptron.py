from sklearn.linear_model import Perceptron as SKlearnPerceptron
from .ClassifieurLineaire import StrategieClassification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from scipy.interpolate import LinearNDInterpolator

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
        self.w = self.perceptron_model.coef_
        self.w_0 = self.perceptron_model.intercept_

    def prediction(self, x):
        """
        Prédit la classe d'une nouvelle donnée x.

        :param classifieur: Une instance de ClassifieurLineaire.
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
        grid_z = self.perceptron_model.predict(grid_tot)
        
        Z = le.transform(grid_z)
        # Remettre les résultats en forme pour le tracé
        Z = Z.reshape(xx.shape)

        

        plt.figure(figsize=(14, 8))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
        plt.scatter(x_train[:, 0], x_train[:, 1], c=t_train_encoded, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())

        plt.title('Frontières de décision - Ensemble d\'Entraienement')
        plt.show()    
        
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
        grid_z = self.perceptron_model.predict(grid_tot)
        
        Z = le.transform(grid_z)
        # Remettre les résultats en forme pour le tracé
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(14, 8))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
        plt.scatter(x_test[:, 0], x_test[:, 1], c=t_test_encoded, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())

        plt.title('Frontières de décision - Données de test')
        plt.show()    
 
    def get_hyperparametres(self):
    
        learning_rate_liste = np.linspace(0.001, 1, 10) #np.array([0.01])
        max_iterations_liste = np.linspace(500, 1500, 10).astype(int)
        penalty_liste = np.array(['l2'])
        
        
        return [ learning_rate_liste,
                 max_iterations_liste,
                 penalty_liste]
    
    def set_hyperparametres(self, hyperparametres_list):
        self.learning_rate = hyperparametres_list[0]
        self.max_iterations = hyperparametres_list[1]
        self.penalty= hyperparametres_list[2]
        
