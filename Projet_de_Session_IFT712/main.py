from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Modele.ClassifieurLineaire import *
from Modele.Perceptron import *
from Modele.SVM import *
from Modele.RandomForest import *
from Modele.RandomForestAvecACP import *
from Modele.AdaBoost import *
from Modele.RechercheHyperparameter.RechercheHyperparameter import *
from Modele.RechercheHyperparameter.SousEchantillonnageAleatoire import *
from Modele.RechercheHyperparameter.ValidationCroisee import *
from Modele.RechercheHyperparameter.BootstrapValidation import *
from Modele.data import TrainData
from sklearn.metrics import accuracy_score

from Projet_de_Session_IFT712.Modele.Reseaux_de_Neurone import Reseaux_Neurones

# Charger un jeu de données pour l'exemple (Leaf dataset)
trainData = TrainData("../leaf-classification/train.csv")
X, y = trainData.data, trainData.leafClass


# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#strategie_perceptron = Perceptron(learning_rate=0.01, max_iterations=1000)
#strategie_perceptron = RandomForest()
#strategie_perceptron = RandomForestAvecACP()
strategie_RN = Reseaux_Neurones((64,48,64))
classifieur = ClassifieurLineaire(strategie_RN)

'''strategie_SVM = SVM(kernel='linear', C=1.0)
classifieur = ClassifieurLineaire(strategie_SVM)'''

# Entraînez le modèle
classifieur.entrainement(X_train, y_train)

# Prédiction sur un exemple de test
exemple_test = X_test[0]
prediction = classifieur.prediction(exemple_test)

predictions = [classifieur.prediction(x) for x in X_test]

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Print or use the accuracy as needed
print(f'Accuracy: {accuracy}')

print("Classe prédite pour l'exemple de test :", prediction)

precision, rappel, f1, _ = classifieur.evaluer(X_test,y_test)

print(f'precision: {precision}')
print(f'rappel: {rappel}')
print(f'f1: {f1}')
# Calcul de l'erreur
erreur = classifieur.erreur(y_test[0], prediction)
print("Erreur de classification :", erreur)


classifieur.afficher_donnees_et_modele(X_train, y_train,X_test,y_test)