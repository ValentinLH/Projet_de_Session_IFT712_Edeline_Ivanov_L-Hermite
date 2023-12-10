from sklearn.svm import SVC
from .ClassifieurLineaire import StrategieClassification
import numpy as np

class SVM(StrategieClassification):
    def __init__(self, kernel='linear', C=1.0):
        """
        Strategie de classification utilisant le svm de scikit-learn.

        :param kernel: specifies the type of kernel to use for the algorithm. Can be linear, poly, rbf, sigmoid or precomputed
        :param C: regularization parameter
        """
        self.kernel = kernel
        self.C = C
        self.svm_modele = None

    def entrainer(self, x_train, t_train):
        """
        Entraine le modele de classification SVM de scikit-learn.

        :param x_train: Les donnees d'entraînement.
        :param t_train: Les étiquettes de classe cibles.
        """
        self.svm_modele = SVC(kernel=self.kernel, C=self.C)
        self.svm_modele.fit(x_train, t_train)
        self.support_vecteurs = self.svm_modele.support_vectors_
        if (self.kernel == "linear"):
            self.w = self.svm_modele.coef_
            self.w_0 = self.svm_modele.intercept_

    def prediction(self, x):
        """
        Predit la classe d'une nouvelle donnee x.

        :param x: La donnee d'entree à classifier.
        :return: 1 si la classe predite est positive, -1 sinon.
        """
        if self.svm_modele is not None:
            return self.svm_modele.predict(x)
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

    def get_hyperparametres(self):
        """
        Renvoie une liste de valeurs que peuvent prendre les hyperparamètres

        :return: Une liste contenant un ensemble de valeur possible pour chaque hyperparamètres
        """
        C_liste = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        noyau_liste = np.array(["linear", "poly", "rbf"])

        return [C_liste, noyau_liste]

        
    def set_hyperparametres(self, hyperparametres_list):
        """
        Met à jour les valeurs des hyperparamètres

        :param hyperparametres_list: liste contenant les nouvelles valeurs des hyperparamètres
        """
        self.C = hyperparametres_list[0]
        self.kernel = hyperparametres_list[1]