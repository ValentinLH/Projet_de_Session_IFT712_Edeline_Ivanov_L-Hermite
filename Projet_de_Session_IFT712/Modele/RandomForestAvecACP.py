from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from .ClassifieurLineaire import StrategieClassification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


"""
Classe de RandomForestAvecACP
Cette classe fonctionne du point de vue technique
Mais elle part d'une idée de faire apprendre le RandomForest uniquement sur le résultat de l'ACP
ce qui ce trouve ne pas etre concluant

"""
class RandomForestAvecACP(StrategieClassification):
    def __init__(self, n_estimators=100, max_depth=None, random_state=None, n_components=2):
        """
        Stratégie de classification utilisant un modèle Random Forest après une ACP.
        Cette classe est expérimentale.

        :param n_estimators: Le nombre d'arbres dans le Random Forest.
        :param max_depth: La profondeur maximale des arbres. (Facultatif)
        :param random_state: Seed pour la reproductibilité. (Facultatif)
        :param n_components: Le nombre de composantes principales à conserver lors de l'ACP.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.random_forest_model = None

    def entrainer(self, x_train, t_train):
        """
        Entraîne le modèle Random Forest après une ACP.

        :param x_train: Les données d'entraînement.
        :param t_train: Les étiquettes de classe cibles.
        """
        # Appliquer l'ACP aux données d'entraînement
        x_train_pca = self.pca.fit_transform(x_train)
        print("Variance expliquée par chaque composante principale :",
              self.pca.explained_variance_ratio_)

        # Entraîner le modèle Random Forest sur les composantes principales
        self.random_forest_model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                                          random_state=self.random_state)
        self.random_forest_model.fit(x_train_pca, t_train)

    def prediction(self, x):
        """
        Prédit la classe d'une nouvelle donnée x.

        :param x: La donnée d'entrée à classifier.
        :return: La prédiction du modèle.
        """
        if self.random_forest_model is not None:
            # Appliquer l'ACP à la nouvelle donnée et prédire avec le modèle Random Forest
            x_pca = self.pca.transform(x)
            return self.random_forest_model.predict(x_pca)
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
            'random_state': self.random_state,
            'n_components': self.n_components
        }

    def afficher(self, x_train, t_train, x_test, t_test, feature_names=None, class_names=None):
        """
        Affiche les résultats de classification pour le modèle Random Forest avec ACP.

        :param x_train: Les données d'entraînement.
        :param t_train: Les étiquettes de classe d'entraînement.
        :param x_test: Les données de test.
        :param t_test: Les étiquettes de classe de test.
        :param feature_names: Noms des caractéristiques (facultatif).
        :param class_names: Noms des classes (facultatif).
        """
        # Appliquer l'ACP aux données de test
        x_test_pca = self.pca.transform(x_test)
        x_train_pca = self.pca.transform(x_train)

        # Utilisez seulement les deux premières composantes principales pour l'affichage
        x_train_pca_subset = x_train_pca[:, :2]
        x_test_pca_subset = x_test_pca[:, :2]

        le = LabelEncoder()
        t_train_encoded = le.fit_transform(t_train)

        # Affichage des données d'entraînement
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(x_train_pca_subset[:, 0], x_train_pca_subset[:, 1], c=t_train_encoded, cmap=plt.cm.Paired,
                    edgecolor='k', s=20)
        plt.title('Training Data (PCA)')

        # Affichage de la frontière de décision pour les deux premières composantes principales
        xx, yy = np.meshgrid(
            np.linspace(np.min(x_train_pca_subset[:, 0]) - 2, np.max(x_train_pca_subset[:, 0]) + 2, 100),
            np.linspace(np.min(x_train_pca_subset[:, 1]) - 2, np.max(x_train_pca_subset[:, 1]) + 2, 100))

        espace = np.c_[xx.ravel(), yy.ravel()]
        Z = self.random_forest_model.predict(espace)
        Z = le.fit_transform(Z)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        t_test_encoded = le.fit_transform(t_test)

        # Affichage des données de test
        plt.subplot(1, 2, 2)
        plt.scatter(x_test_pca_subset[:, 0], x_test_pca_subset[:, 1], c=t_test_encoded, cmap=plt.cm.Paired,
                    edgecolor='k', s=20)
        plt.title('Testing Data (PCA)')

        # Affichage de la frontière de décision pour les deux premières composantes principales des données de test
        xx, yy = np.meshgrid(np.linspace(np.min(x_test_pca_subset[:, 0]) - 2, np.max(x_test_pca_subset[:, 0]) + 2, 100),
                             np.linspace(np.min(x_test_pca_subset[:, 1]) - 2, np.max(x_test_pca_subset[:, 1]) + 2, 100))

        espace = np.c_[xx.ravel(), yy.ravel()]
        Z = self.random_forest_model.predict(espace)
        Z = le.fit_transform(Z)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Affichage final
        plt.show()
