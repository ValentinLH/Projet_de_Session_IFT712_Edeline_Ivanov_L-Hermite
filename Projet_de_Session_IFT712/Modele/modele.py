from abc import ABC, abstractmethod

class Modele(ABC):

    @abstractmethod
    def entrainer(self, donnees):
        pass

    @abstractmethod
    def predire(self, donnees):
        pass

    @abstractmethod
    def erreur(self):
        pass

    @abstractmethod
    def afficher(self):
        pass