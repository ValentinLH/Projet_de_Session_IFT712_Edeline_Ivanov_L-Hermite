from sklearn.svm import SVC
from .ClassifieurLineaire import StrategieClassification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from scipy.interpolate import LinearNDInterpolator

class SVM(StrategieClassification):
    def __init__(self, kernel='linear', C=1.0):
        """
        Strategie de classification utilisant le svm de scikit-learn.

        :param kernel: specifies the type of kernel to use for the algorithm. Can be linear, poly, rbf, sigmoid or precomputed
        :param C: regularization parameter
        """
        self.kernel = kernel
        self.C = C
        self.svm_model = None

    def entrainer(self, x_train, t_train):
        """
        Entraine le modele de classification SVM de scikit-learn.

        :param x_train: Les donnees d'entraînement.
        :param t_train: Les étiquettes de classe cibles.
        """
        self.svm_model = SVC(kernel=self.kernel, C=self.C)
        self.svm_model.fit(x_train, t_train)
        self.support_vectors = self.svm_model.support_vectors_
        self.w = self.svm_model.coef_
        self.w_0 = self.svm_model.intercept_

    def prediction(self, x):
        """
        Predit la classe d'une nouvelle donnee x.

        :param x: La donnee d'entree à classifier.
        :return: 1 si la classe predite est positive, -1 sinon.
        """
        if self.svm_model is not None:
            return self.svm_model.predict(x)
        return 0
    
    def parametres(self):
        """
        Retourne les parametres du classifieur

        :return: 1
        """
        return self.w_0, self.w

    def erreur(self, t, prediction):
        """
        Calcule l'erreur de classification.

        :param t: etiquette de classe.
        :param prediction: La prediction du modele.
        :return: 1 si l'erreur est commise, 0 sinon.
        """
        return 1 if t != prediction else 0

    def afficher(self, x_train, t_train, x_test, t_test):
        """
        Affiche les données dans un espace à deux dimensions

        :param x_train: donnees d'entrainement
        :param t_train: etiquettes associees aux donnees d'entrainement
        :param x_test: donnees de test
        :param t_train: etiquettes associees aux donnees de test
        """

        le = LabelEncoder()
        t_train_encoded = le.fit_transform(t_train)
        t_test_encoded = le.transform(t_test)

        h = 0.05
        x_min, x_max = x_train[:, 0].min() - .5, x_train[:, 0].max() + .5
        y_min, y_max = x_train[:, 1].min() - .5, x_train[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Utiliser LinearNDInterpolator pour interpoler les donnees
        points = np.column_stack((x_train[:, 0], x_train[:, 1]))
        values = x_train[:,2:]
        
        interpolator = LinearNDInterpolator(points, values)
        grid_xy = np.c_[xx.ravel(), yy.ravel()]
        grid_dim = interpolator(grid_xy)
        grid_tot = np.c_[grid_xy,grid_dim]
        grid_tot[np.isnan(grid_tot)] = 0
        grid_z = self.svm_model.predict(grid_tot)
        
        Z = le.transform(grid_z)
        # Remettre les resultats en forme pour le trace
        Z = Z.reshape(xx.shape) 

        plt.figure(0)

        plt.figure(figsize=(14, 8))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
        plt.scatter(x_train[:, 0], x_train[:, 1], c=t_train_encoded, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())

        plt.title('Frontieres de decision - donnees d\'Entrainement')

        h = 0.05
        x_min, x_max = x_test[:, 0].min() - .5, x_test[:, 0].max() + .5
        y_min, y_max = x_test[:, 1].min() - .5, x_test[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Utiliser LinearNDInterpolator pour interpoler les donnees
        points = np.column_stack((x_test[:, 0], x_test[:, 1]))
        values = x_test[:,2:]
        
        interpolator = LinearNDInterpolator(points, values)
        grid_xy = np.c_[xx.ravel(), yy.ravel()]
        grid_dim = interpolator(grid_xy)
        grid_tot = np.c_[grid_xy,grid_dim]
        grid_tot[np.isnan(grid_tot)] = 0
        grid_z = self.svm_model.predict(grid_tot)
        
        Z = le.transform(grid_z)
        # Remettre les resultats en forme pour le trace
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

        plt.title('Frontieres de decision - Donnees de test')

        plt.show()