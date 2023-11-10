from sklearn.ensemble import RandomForestClassifier
from .ClassifieurLineaire import StrategieClassification
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree


class RandomForest(StrategieClassification):
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        """
        Stratégie de classification utilisant un modèle Random Forest de scikit-learn.

        :param n_estimators: Le nombre d'arbres dans le Random Forest.
        :param max_depth: La profondeur maximale des arbres. (Facultatif)
        :param random_state: Seed pour la reproductibilité. (Facultatif)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.random_forest_model = None

    def entrainer(self, x_train, t_train):
        """
        Entraîne le modèle Random Forest de scikit-learn.

        :param x_train: Les données d'entraînement.
        :param t_train: Les étiquettes de classe cibles.
        """
        self.random_forest_model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)
        self.random_forest_model.fit(x_train, t_train)

    def prediction(self, x):
        """
        Prédit la classe d'une nouvelle donnée x.

        :param x: La donnée d'entrée à classifier.
        :return: La prédiction du modèle.
        """
        if self.random_forest_model is not None:
            return self.random_forest_model.predict([x])[0]
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
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }

    
    def afficher(self, x_train, t_train, x_test, t_test, feature_names=None, class_names=None):
        """
        Affiche les résultats de classification pour le modèle Random Forest.

        :param x_train: Les données d'entraînement.
        :param t_train: Les étiquettes de classe d'entraînement.
        :param x_test: Les données de test.
        :param t_test: Les étiquettes de classe de test.
        :param feature_names: Noms des caractéristiques (facultatif).
        :param class_names: Noms des classes (facultatif).
        """
        if not feature_names:
            feature_names = [f'Feature {i}' for i in range(x_train.shape[1])]
        if not class_names:
            class_names = [f'Class {i}' for i in range(len(np.unique(t_train)))]

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
        tree.plot_tree(self.random_forest_model.estimators_[0],
                    feature_names=feature_names,
                    class_names=class_names,
                    filled=True)
        fig.savefig('rf_individualtree.png')
        plt.show()

        return

        # Utilisez seulement les deux premières caractéristiques pour l'affichage
        x_train_subset = x_train[:, :2]
        x_test_subset = x_test[:, :2]

        # Affichage des données d'entraînement
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(x_train_subset[:, 0], x_train_subset[:, 1], c=t_train, cmap=plt.cm.Paired, edgecolor='k', s=20)
        plt.title('Training Data')

        # Affichage de la frontière de décision
        xx, yy = np.meshgrid(np.linspace(np.min(x_train_subset[:, 0]) - 2, np.max(x_train_subset[:, 0]) + 2, 100),
                            np.linspace(np.min(x_train_subset[:, 1]) - 2, np.max(x_train_subset[:, 1]) + 2, 100))
        Z = self.random_forest_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Affichage des données de test
        plt.subplot(1, 2, 2)
        plt.scatter(x_test_subset[:, 0], x_test_subset[:, 1], c=t_test, cmap=plt.cm.Paired, edgecolor='k', s=20)
        plt.title('Testing Data')

        # Affichage de la frontière de décision
        xx, yy = np.meshgrid(np.linspace(np.min(x_test_subset[:, 0]) - 2, np.max(x_test_subset[:, 0]) + 2, 100),
                            np.linspace(np.min(x_test_subset[:, 1]) - 2, np.max(x_test_subset[:, 1]) + 2, 100))
        Z = self.random_forest_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        plt.show()
