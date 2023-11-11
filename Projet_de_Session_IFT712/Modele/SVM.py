from sklearn.svm import SVC
from .ClassifieurLineaire import StrategieClassification
import numpy as np
import matplotlib.pyplot as plt

class SVM(StrategieClassification):
    def __init__(self, kernel='linear', C=1.0):
        self.kernel = kernel
        self.C = C
        self.svm_model = None

    def entrainer(self, x_train, t_train):
        self.svm_model = SVC(kernel=self.kernel, C=self.C)
        self.svm_model.fit(x_train, t_train)
        self.support_vectors = self.svm_model.support_vectors_
        self.w = self.svm_model.coef_[0]
        self.w_0 = -self.svm_model.intercept_[0]

    def prediction(self, x):
        if self.svm_model is not None:
            return self.svm_model.predict([x])[0]
        return 0  # Valeur par défaut si le modèle n'est pas encore entraîné

    def parametres(self):
        return self.w_0, self.w

    def erreur(self, t, prediction):
        return 1 if t != prediction else 0

    def afficher(self, x_train, t_train, x_test, t_test):

        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=t_train * 100 + 20, c=t_train)

        if self.kernel == 'linear':
            pente = -self.w[0] / self.w[1]
            xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
            yy = pente * xx - self.w_0 / self.w[1]
            plt.plot(xx, yy, label='Decision Boundary')

        plt.scatter(self.support_vectors[:, 0], self.support_vectors[:, 1], s=200, facecolors='none', edgecolors='r', label='Support Vectors')

        plt.title('Training data')
        plt.legend()

        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=t_test * 100 + 20, c=t_test)

        if self.kernel == 'linear':
            pente = -self.w[0] / self.w[1]
            xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
            yy = pente * xx - self.w_0 / self.w[1]
            plt.plot(xx, yy, label='Decision Boundary')

        plt.title('Testing data')
        plt.legend()

        plt.show()