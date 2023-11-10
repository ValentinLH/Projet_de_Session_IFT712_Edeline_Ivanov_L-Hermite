from abc import ABC, abstractmethod

# Créez une classe abstraite pour la stratégie
class StrategieClassification(ABC):   
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
    def erreur(self,t, prediction):
        pass
    
    @abstractmethod
    def afficher(self, x_train, t_train, x_test, t_test):
        pass
    

# Classe ClassifieurLineaire avec les méthodes nécessaires pour travailler avec des stratégies de classification
class ClassifieurLineaire:
    def __init__(self, strategie : StrategieClassification):
        """
        Algorithmes de classification lineaire

        La classe prend  une instance de la stratégie de classification.
        """
        self.w = None
        self.w_0 = None
        self.strategie = strategie

    def entrainement(self, x_train, t_train):
        # Utilisez la stratégie pour l'entraînement
        self.strategie.entrainer(x_train, t_train)
        self.w_0, self.w = self.strategie.parametres()

    def prediction(self, x):
        # Utilisez la stratégie pour la prédiction
        return self.strategie.prediction(x)

    def erreur(self, t, prediction):
        # Utilisez la stratégie pour calculer l'erreur
        return self.strategie.erreur(t, prediction)

    def afficher_donnees_et_modele(self, x_train, t_train, x_test, t_test):
        """
        afficher les donnees et le modele

        x_train, t_train : donnees d'entrainement
        x_test, t_test : donnees de test
        """
        self.strategie.afficher(x_train, t_train, x_test, t_test)

    def parametres(self):
        """
        Retourne les paramètres du modèle
        """
        return self.strategie.parametres()
