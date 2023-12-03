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

# Charger un jeu de données pour l'exemple (Leaf dataset)
trainData = TrainData("leaf-classification/train.csv")
X, y = trainData.data, trainData.leafClass


# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

strategie_perceptron = Perceptron(learning_rate=0.01, max_iterations=1000)
#strategie_perceptron = RandomForest()  
#strategie_perceptron = RandomForestAvecACP()
classifieur = ClassifieurLineaire(strategie_perceptron)

#strategie_Adaboost = AdaBoost(n_estimators=200, learning_rate=0.01, random_state=0, algorithm="SAMME.R", max_depth_tree_classifieur=3)
#classifieur = ClassifieurLineaire(strategie_Adaboost)


#Recherche d'hyperparamètres
#Validation croisée

#stategie_hyper_parametre = BootstrapValidation(2,10)
#Recherche = RechercheHyperparameter(stategie_hyper_parametre)
#Recherche.recherche(classifieur, X, y)


#Sous echantillonnage aléatoire
'''stategie_hyper_parametre = SousEchantillonnage(10, 0.2)
Recherche = RechercheHyperparameter(stategie_hyper_parametre)
Recherche.recherche(classifieur, X, y)'''


'''strategie_SVM = SVM(kernel='linear', C=1.0)
classifieur = ClassifieurLineaire(strategie_SVM)'''

# Entraînez le modèle
classifieur.entrainement(X_train, y_train)

# Prédiction sur un exemple de test
exemple_test = X_test[0]
prediction = classifieur.prediction([exemple_test])

predictions = classifieur.prediction(X_test) 

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Print or use the accuracy as needed
print(f'Accuracy: {accuracy}')


precision, rappel, f1, _ = classifieur.evaluer(X_test,y_test)

print(f'precision: {precision}')
print(f'rappel: {rappel}')
print(f'f1: {f1}')
# Calcul de l'erreur

classifieur.afficher_donnees_et_modele(X_train, y_train,X_test,y_test)