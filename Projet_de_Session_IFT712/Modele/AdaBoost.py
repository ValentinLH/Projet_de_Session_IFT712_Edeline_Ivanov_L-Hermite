from .ClassifieurLineaire import StrategieClassification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from scipy.interpolate import LinearNDInterpolator

class AdaBoost(StrategieClassification):
    def __init__(self, n_estimators=50, learning_rate=0.01, random_state=0, algorithm="SAMME.R", max_depth_tree_classifieur=1):
        """
        Strategie de classification utilisant le svm de scikit-learn.

        :param n_estimators: nombre de modele utilisé
        :param learning_rate: taus d'apprentissage, correspond au poids donné pour chaque modele 
        :param random_state: indique le type d'aléatoire donnée pour chaque modèle
        :param algorithm: type d'algorithme utilisé pour executer AdaBoost, peut être "SAMME" ou "SAMME.R"
        :param max_depth_tree_classifieur: profondeur de l'arbre de décision utilisé par l'lagorithme
        """

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.algorithm = algorithm
        self.max_depth_tree_classifieur = max_depth_tree_classifieur
        self.adaboost_model = None

    def entrainer(self, x_train, t_train):
        """
        Entraine le modele de classification Ada boost de scikit-learn.

        :param x_train: Les donnees d'entraînement.
        :param t_train: Les étiquettes de classe cibles.
        """
        tree_classifier = DecisionTreeClassifier(max_depth=self.max_depth_tree_classifieur)
        self.adaboost_model = AdaBoostClassifier(tree_classifier, n_estimators=self.n_estimators, learning_rate=self.learning_rate, 
                                                 random_state=self.random_state, algorithm=self.algorithm)
        self.adaboost_model.fit(x_train, t_train)

    def prediction(self, x):
        """
        Predit la classe d'une nouvelle donnee x.

        :param x: La donnee d'entree à classifier.
        :return: 1 si la classe predite est positive, -1 sinon.
        """
        if self.adaboost_model is not None:
            return self.adaboost_model.predict(x)
        return 0
    
    def parametres(self):
        """
        Retourne les parametres du classifieur

        :return: dictionnaire composé des parametres associé à leur valeur
        """
        return {'n_estimators': self.n_estimators, 'learning_rate': self.learning_rate, 'random_state': self.random_state,
                'algorithm': self.algorithm, "max_depth_tree_classifieur": self.max_depth_tree_classifieur}

    def erreur(self, t, prediction):
        """
        Calcule l'erreur de classification.

        :param t: etiquette de classe.
        :param prediction: La prediction du modele.
        :return: 1 si l'erreur est commise, 0 sinon.
        """
        return 1 if t != prediction else 0
    
    def get_hyperparametres(self):
        '''estimator_liste = np.arange(50, 501, 50, dtype=int)
        learning_rate_liste = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5])
        random_state_liste = np.array([0, 1, 5, 10, 20, 50, 75, 100])
        algorithm_liste = np.array(["SAMME", "SAMME.R"])
        depth_liste = np.arange(1, 9, 1, dtype=int)'''

        estimator_liste = np.array([50])
        learning_rate_liste = np.array([0.001])
        random_state_liste = np.array([75])
        algorithm_liste = np.array(["SAMME", "SAMME.R"])
        depth_liste = np.array([5])

        return [estimator_liste,
                         learning_rate_liste,
                         random_state_liste,
                         algorithm_liste,
                         depth_liste]
    
    def set_hyperparametres(self, hyperparametres_list):
        self.n_estimators = hyperparametres_list[0]
        self.learning_rate = hyperparametres_list[1]
        self.random_state = hyperparametres_list[2]
        self.algorithm = hyperparametres_list[3]
        self.max_depth_tree_classifieur = hyperparametres_list[4]

    
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