import torch
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
from Modele.Convolutional_Neural_Network import Net

# Charger un jeu de données pour l'exemple (Leaf dataset)
trainData = TrainData("leaf-classification/train.csv")
X, y = trainData.data, trainData.leafClass

train_loader = trainData.read_image()
trainData.imshow()

net = Net()
classifieur = ClassifieurLineaire(net)

dataiter = torch.utils.data.DataLoader.__iter__((train_loader))
images, labels = dataiter.__next__()

X_original = torch.from_numpy(X[1:].astype(np.float32))  # Convertit en tenseur PyTorch

# Créez une matrice de zéros avec la forme souhaitée (989, 224 - 192)
zeros_matrix = torch.zeros(X_original.size(0), images.size()[3] - X_original.size(1))

# Concaténez la matrice originale et la matrice de zéros le long de la deuxième dimension
X_resized = torch.cat([X_original, zeros_matrix], dim=1)


X_resized = X_resized.unsqueeze(1)

# Concaténez X_resized à images le long de la dernière dimension
images = torch.cat([images, X_resized.unsqueeze(-1)], dim=3)
images = torch.cat([images, X_resized.unsqueeze(-1)], dim=3)
images = torch.cat([images, X_resized.unsqueeze(-1)], dim=3)
images = torch.cat([images, X_resized.unsqueeze(-1)], dim=3)




X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)



"""
#strategie_perceptron = Perceptron(learning_rate=0.01, max_iterations=1000)
strategie_perceptron = RandomForest()  
#strategie_perceptron = RandomForestAvecACP()
classifieur = ClassifieurLineaire(strategie_perceptron)
"""

#strategie_Adaboost = AdaBoost(n_estimators=200, learning_rate=0.01, random_state=0, algorithm="SAMME.R", max_depth_tree_classifieur=3)
#classifieur = ClassifieurLineaire(strategie_Adaboost)


#Recherche d'hyperparamètres
#Validation croisée
"""
stategie_hyper_parametre = ValidationCroisee(10)
Recherche = RechercheHyperparameter(stategie_hyper_parametre)
Recherche.recherche(classifieur, X, y)
"""

#Sous echantillonnage aléatoire
'''stategie_hyper_parametre = SousEchantillonnage(10, 0.2)
Recherche = RechercheHyperparameter(stategie_hyper_parametre)
Recherche.recherche(classifieur, X, y)'''


'''strategie_SVM = SVM(kernel='linear', C=1.0)
classifieur = ClassifieurLineaire(strategie_SVM)'''

# Entraînez le modèle
classifieur.entrainement(X_train, y_train)

# Prédiction sur un exemple de test
#predictions = [classifieur.prediction(x) for x in X_test]
predictions = classifieur.prediction(X_test)
print("classe predict = ", predictions)

#predictions = classifieur.prediction(X_test) 
"""
ok  =[]# [classifieur.erreur(labels[0]),torch.zeros(64)[predictions[i]]+=1) for i in range(len(test))]
for i in range(len(predictions)) :
    temp = torch.zeros(99)
    temp[predictions[i]]+=1
    ok.append(temp)

err = classifieur.erreur(y_test,torch.stack(ok))

print('erreur = ', err)
"""
_, y_test_pred = torch.max(y_test, 1)
y_test_pred_list = y_test_pred.tolist()

_, y_train_pred = torch.max(y_train, 1)
y_train_pred_list = y_train_pred.tolist()


precision, rappel, f1, _ = classifieur.evaluer(X_train, y_train_pred_list)

print(f'Metrique entrainement : ')
print(f'precision: {precision}')
print(f'rappel: {rappel}')
print(f'f1: {f1}')

precision, rappel, f1, _ = classifieur.evaluer(X_test,y_test_pred_list)

print(f'\n\n########\nMetrique Test : ')
print(f'precision: {precision}')
print(f'rappel: {rappel}')
print(f'f1: {f1}')

#classifieur.afficher_donnees_et_modele(X_train, y_train_pred_list,X_test,y_test_pred_list)


#precision, rappel, f1, _ = classifieur.evaluer(X_test,y_test)

#print(f'precision: {precision}')
#print(f'rappel: {rappel}')
#print(f'f1: {f1}')
# Calcul de l'erreur

#classifieur.afficher_donnees_et_modele(X_train, y_train,X_test,y_test)

# Normalisation des données
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
#
# # Diviser les données en ensembles d'entraînement et de test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# strategie_perceptron = Perceptron(learning_rate=0.01, max_iterations=1000)
# #strategie_perceptron = RandomForest()
# #strategie_perceptron = RandomForestAvecACP()
# classifieur = ClassifieurLineaire(strategie_perceptron)
#
# '''strategie_SVM = SVM(kernel='linear', C=1.0)
# classifieur = ClassifieurLineaire(strategie_SVM)'''
#
# # Entraînez le modèle
# classifieur.entrainement(X_train, y_train)
#
# # Prédiction sur un exemple de test
# exemple_test = X_test[0]
# prediction = classifieur.prediction(exemple_test)
#
# predictions = [classifieur.prediction(x) for x in X_test]
#
# # Calculate accuracy
# accuracy = accuracy_score(y_test, predictions)
#
# # Print or use the accuracy as needed
# print(f'Accuracy: {accuracy}')
#
# print("Classe prédite pour l'exemple de test :", prediction)
#
# # Calcul de l'erreur
# erreur = classifieur.erreur(y_test[0], prediction)
# print("Erreur de classification :", erreur)
#
#
