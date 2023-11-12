import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from Modele.ClassifieurLineaire import *
from Modele.Perceptron import *
from Modele.SVM import *
from Modele.RandomForest import *

# Charger un jeu de données pour l'exemple (Iris dataset)
data = load_iris()
X, y = data.data, data.target

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

strategie_perceptron = Perceptron(learning_rate=0.01, max_iterations=1000)
#strategie_perceptron = RandomForest()
classifieur = ClassifieurLineaire(strategie_perceptron)

'''strategie_SVM = SVM(kernel='linear', C=1.0)
classifieur = ClassifieurLineaire(strategie_SVM)'''

# Entraînez le modèle
classifieur.entrainement(X_train, y_train)

# Prédiction sur un exemple de test
exemple_test = X_test[0]
prediction = classifieur.prediction(exemple_test)

print("Classe prédite pour l'exemple de test :", prediction)

# Calcul de l'erreur
erreur = classifieur.erreur(y_test[0], prediction)
print("Erreur de classification :", erreur)


classifieur.afficher_donnees_et_modele(X_train, y_train,X_test,y_test)