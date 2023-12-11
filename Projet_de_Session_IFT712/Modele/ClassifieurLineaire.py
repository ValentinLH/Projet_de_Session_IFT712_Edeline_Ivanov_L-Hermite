from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from sklearn.calibration import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch


# Créez une classe abstraite pour la stratégie
class StrategieClassification(ABC):
    """
    Classe abstraite représentant le squelette de nos méthode de classification.

    Méthodes abstraites à implémenter par les classes concrètes :
        - entrainer(x_train, t_train): Entraîne le modèle de classification sur les données d'entraînement.
        - prediction(x): Prédit les étiquettes de classe pour un ensemble de données.
        - parametres(): Retourne les paramètres actuels du modèle de classification.
        - erreur(t, prediction): Calcule l'erreur entre les étiquettes réelles et les prédictions du modèle.
        - afficher(x_train, t_train, x_test, t_test): Affiche des informations sur les performances du modèle.

    Les classes concrètes doivent fournir des implémentations pour ces méthodes
    afin de définir une stratégie de classification spécifique.

    """
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
    def erreur(self, t, prediction):
        pass

    @abstractmethod
    def afficher(self, x_train, t_train, x_test, t_test):
        pass


# Classe ClassifieurLineaire avec les méthodes nécessaires pour travailler avec des stratégies de classification
class ClassifieurLineaire:
    def __init__(self, strategie: StrategieClassification):
        """
        Algorithmes de classification lineaire

        La classe prend une instance de la stratégie de classification.
        """
        self.strategie = strategie

    def entrainement(self, x_train, t_train):
        """
        Entraîne le modèle à l'aide de la stratégie de classification.

        Parametres:
        - x_train (array): Données d'entraînement.
        - t_train (array): Étiquettes d'entraînement.
        """
        self.strategie.entrainer(x_train, t_train)

    def prediction(self, x):
        """
        Effectue des prédictions à l'aide de la stratégie de classification.

        Parametres:
        - x (array): Données pour les prédictions.

        Retourne:
        - array: Prédictions du modèle.
        """
        if not isinstance(x, (list, np.ndarray, torch.Tensor)):
            x = [x]

        return self.strategie.prediction(x)

    def erreur(self, t, prediction):
        """
        Calcule l'erreur entre les étiquettes réelles et les prédictions.

        Parameters:
        - t (array): Étiquettes réelles.
        - prediction (array): Prédictions du modèle.

        Returns:
        - list: Erreurs du modèle.
        """
        return self.strategie.erreur(t, prediction)

    def afficher_donnees_et_modele(self, x_train, t_train, x_test, t_test):
        """
        Affiche les informations que le modele peut souhaiter afficher et
        les frontières de décision pour l'ensemble d'entraînement et de test.

        x_train, t_train : donnees d'entrainement
        x_test, t_test : donnees de test
        """
        self.strategie.afficher(x_train, t_train, x_test, t_test)
        self.afficher(x_train, t_train, x_test, t_test)

    def parametres(self):
        """
        Retourne les paramètres du modèle
        
        Retourne:
        - Paramètres du modèle (Dépend du modele de classification).
        """
        return self.strategie.parametres()

    def get_hyperparametres(self):
        """
        Retourne les hyperparamètres du modèle.
        Dans l'optique du recherche d'hyperparamètres optimaux
        
        Retourne:
        - Hyperparamètres du modèle.
        """
        return self.strategie.get_hyperparametres()

    def set_hyperparametres(self, hyperparametres_list):
        """
        Définit les hyperparamètres du modèle.

        Parametres:
        - hyperparametres_list (list): Liste des hyperparamètres.
        """
        self.strategie.set_hyperparametres(hyperparametres_list)

    def afficher(self, x_train, t_train, x_test, t_test):
        """
        Affiche les frontières de décision pour l'ensemble d'entraînement et de test.

        Parameters:
        - x_train (array): Données d'entraînement.
        - t_train (array): Étiquettes d'entraînement.
        - x_test (array): Données de test.
        - t_test (array): Étiquettes de test.
        """
        le = LabelEncoder()
        t_train_encoded = le.fit_transform(t_train)
        t_test_encoded = le.transform(t_test)

        h = 0.05
        x_min, x_max = x_train[:, 0].min() - .5, x_train[:, 0].max() + .5
        y_min, y_max = x_train[:, 1].min() - .5, x_train[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Utiliser LinearNDInterpolator pour interpoler les données
        points = np.column_stack((x_train[:, 0], x_train[:, 1]))
        values = x_train[:, 2:]

        interpolator = LinearNDInterpolator(points, values)
        grid_xy = np.c_[xx.ravel(), yy.ravel()]
        grid_dim = interpolator(grid_xy)
        grid_tot = np.c_[grid_xy, grid_dim]
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
        values = x_test[:, 2:]

        interpolator = LinearNDInterpolator(points, values)
        grid_xy = np.c_[xx.ravel(), yy.ravel()]
        grid_dim = interpolator(grid_xy)
        grid_tot = np.c_[grid_xy, grid_dim]
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

        # Fonction pour évaluer le modèle

    def evaluer(self, X, y):
        """
        Évalue le modèle en affichant la matrice de confusion et en renvoyant les métriques de performance.

        Parametres:
        - X (array): Données pour l'évaluation.
        - y (array): Étiquettes pour l'évaluation.

        Retourne:
        - precision (float): Précision pondérée.
        - rappel (float): Rappel pondéré.
        - f1 (float): Score F1 pondéré.
        - matrice_confusion (array): Matrice de confusion.
        """
        predictions = self.prediction(X)
        precision = precision_score(y, predictions, average='weighted')
        rappel = recall_score(y, predictions, average='weighted')
        f1 = f1_score(y, predictions, average='weighted')
        matrice_confusion = confusion_matrix(y, predictions)
        # Rapport de classification
        # class_report = classification_report(X, y)

        # Tracer la matrice de confusion
        plt.imshow(matrice_confusion, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matrice de Confusion')
        plt.colorbar()
        plt.xlabel('Vraies étiquettes')
        plt.ylabel('Étiquettes prédites')
        plt.show()

        return precision, rappel, f1, matrice_confusion
