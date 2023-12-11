from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from .ClassifieurLineaire import StrategieClassification
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree


class RandomForest(StrategieClassification):
    def __init__(self, n_estimators=100, criterion="gini", max_depth=None, random_state=None):
        """
        Stratégie de classification utilisant un modèle Random Forest de scikit-learn.

        :param n_estimators: Le nombre d'arbres dans le Random Forest.
        :param max_depth: La profondeur maximale des arbres. (Facultatif)
        :param random_state: Seed pour la reproductibilité. (Facultatif)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.criterion = criterion
        self.random_forest_model = None

    def entrainer(self, x_train, t_train):
        """
        Entraîne le modèle Random Forest de scikit-learn.

        :param x_train: Les données d'entraînement.
        :param t_train: Les étiquettes de classe cibles.
        """
        self.random_forest_model = RandomForestClassifier(criterion=self.criterion, n_estimators=self.n_estimators,
                                                          max_depth=self.max_depth, random_state=self.random_state)
        self.random_forest_model.fit(x_train, t_train)

    def prediction(self, x):
        """
        Prédit la classe d'une nouvelle donnée x.

        :param x: La donnée d'entrée à classifier.
        :return: La prédiction du modèle.
        """
        if self.random_forest_model is not None:
            return self.random_forest_model.predict(x)
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

    def afficher(self, x_entrainement, t_entrainement, x_test, t_test, noms_attributs=None, noms_classes=None):
        """
        Affiche les résultats de classification pour le modèle Random Forest.
        On peut y voir l'arbre de décisions qui est aussi sauvegarder au format png

        :param x_train: Les données d'entraînement.
        :param t_train: Les étiquettes de classe d'entraînement.
        :param x_test: Les données de test.
        :param t_test: Les étiquettes de classe de test.
        :param feature_names: Noms des caractéristiques (facultatif).
        :param class_names: Noms des classes (facultatif).
        """
        # Si les noms de caractéristiques ne sont pas fournis, utilisez des noms génériques.
        if not noms_attributs:
            noms_attributs = [f'Attribut {i}' for i in range(x_entrainement.shape[1])]

        # Si les noms de classe ne sont pas fournis,
        # utilisez des noms génériques basés sur les étiquettes d'entraînement.
        if not noms_classes:
            noms_classes = [f'Classe {i}' for i in range(len(np.unique(t_entrainement)))]

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
        tree.plot_tree(self.random_forest_model.estimators_[0],
                       feature_names=noms_attributs,
                       class_names=noms_classes,
                       filled=True)
        fig.savefig('arbre_de_décision_random_forest.png')
        plt.show()

        # Utilisez seulement les deux premières caractéristiques pour l'affichage
        x_entrainement_sous_ensemble = x_entrainement[:, :2]
        x_test_sous_ensemble = x_test[:, :2]

        le = LabelEncoder()
        x_entrainement_encode = le.fit_transform(t_entrainement)

        # Affichage des données d'entraînement
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(x_entrainement_sous_ensemble[:, 0], x_entrainement_sous_ensemble[:, 1], c=x_entrainement_encode,
                    cmap=plt.cm.Paired, edgecolor='k',
                    s=20)
        plt.title('Training Data')

        t_test_encoded = le.fit_transform(t_test)

        # Affichage des données de test
        plt.subplot(1, 2, 2)
        plt.scatter(x_test_sous_ensemble[:, 0], x_test_sous_ensemble[:, 1], c=t_test_encoded,
                    cmap=plt.cm.Paired, edgecolor='k', s=20)
        plt.title('Testing Data')

        # Affichage final
        plt.show()

    def get_hyperparametres(self):
        """
        Renvoie une liste de valeurs que peuvent prendre les hyperparamètres

        :return: Une liste contenant un ensemble de valeur possible pour chaque hyperparamètres
        """
        n_estimators_liste = np.linspace(70, 150, 10).astype(int)
        criterion_liste = np.array(["gini", "log_loss", "entropy"])

        return [n_estimators_liste,
                criterion_liste,
                ]

    def set_hyperparametres(self, hyperparametres_list):
        """
        Met à jour les valeurs des hyperparamètres

        :param hyperparametres_list: liste contenant les nouvelles valeurs des hyperparamètres
        """
        self.n_estimators = hyperparametres_list[0]
        self.criterion = hyperparametres_list[1]
