from sklearn.linear_model import Perceptron as SKlearnPerceptron
from .ClassifieurLineaire import StrategieClassification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder

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
            return self.perceptron_model.predict([x])[0]
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

        # Encode training labels
        t_train_encoded = le.fit_transform(t_train)
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=(t_train_encoded * 100 + 20).astype(int), c=t_train_encoded)

        pente = -self.perceptron_model.coef_[0, 0] / self.perceptron_model.coef_[0, 1]
        xx_train = np.linspace(np.min(x_train[:, 0]) - 2, np.max(x_train[:, 0]) + 2, num=99)

        yy_train = pente * xx_train - self.perceptron_model.intercept_ / self.perceptron_model.coef_[0, 1]
        plt.plot(xx_train, yy_train)
        plt.title('Training data')

        # Encode testing labels
        t_test_encoded = le.transform(t_test)
        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=(t_test_encoded * 100 + 20).astype(int), c=t_test_encoded)

        pente = -self.perceptron_model.coef_[0, 0] / self.perceptron_model.coef_[0, 1]
        xx_test = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2, num=99)
        yy_test = pente * xx_test - self.perceptron_model.intercept_ / self.perceptron_model.coef_[0, 1]
        plt.plot(xx_test, yy_test)
        plt.title('Testing data')

        plt.show()

        return
        le = LabelEncoder()

        # Encode training labels
        t_train_encoded = le.fit_transform(t_train)
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=(t_train_encoded * 100 + 20).astype(int), c=t_train_encoded)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Training data')

        # Encode testing labels
        t_test_encoded = le.transform(t_test)
        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=(t_test_encoded * 100 + 20).astype(int), c=t_test_encoded)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Testing data')

        plt.show()
  
        
        return
        le = LabelEncoder()

        # Encode training labels
        t_train_encoded = le.fit_transform(t_train)
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=(t_train_encoded * 100 + 20), c=t_train_encoded)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Training data')

        # Encode testing labels
        t_test_encoded = le.transform(t_test)
        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=(t_test_encoded * 100 + 20), c=t_test_encoded)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Testing data')

        plt.show()
        
"""
    def afficher(self, x_train, t_train, x_test=None, t_test=None, feature_names=None, class_names=None):
        if not feature_names:
            feature_names = [f'Feature {i}' for i in range(x_train.shape[1])]
        if not class_names:
            class_names = [f'Class {i}' for i in range(len(np.unique(t_train)))]

        # Create a mesh grid to plot decision boundaries
        h = .02  # step size in the mesh
        x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
        y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict the labels for each point in the mesh grid
        Z = self.perceptron_model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Reshape the predictions and plot the decision boundaries
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot the data points
        le = LabelEncoder()
        t_encoded = le.fit_transform(t_train)
        plt.scatter(x_train[:, 0], x_train[:, 1], c=t_encoded, edgecolors='k', cmap=plt.cm.Paired)

        # Labeling
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title('Perceptron Decision Boundaries')

        # Add a legend
        unique_classes = np.unique(t_train)
        class_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.Paired.colors[le.transform([cls])[0]],
                                markersize=10, label=class_names[le.transform([cls])[0]]) for cls in unique_classes]
        plt.legend(handles=class_handles)

        plt.show()
"""