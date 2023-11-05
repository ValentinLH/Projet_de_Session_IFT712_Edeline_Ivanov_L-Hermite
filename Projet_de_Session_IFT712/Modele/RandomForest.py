from sklearn.ensemble import RandomForestClassifier
from .ClassifieurLineaire import StrategieClassification

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
