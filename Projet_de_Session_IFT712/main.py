import torch
from sklearn.model_selection import train_test_split

from Modele.ClassifieurLineaire import *
from Modele.Perceptron import *
from Modele.SVM import *
from Modele.RandomForest import *
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

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

classifieur.entrainement(X_train,y_train)
predictions = [classifieur.prediction(x) for x in X_test]
print("classe predict = ", predictions)

ok  =[]# [classifieur.erreur(labels[0]),torch.zeros(64)[predictions[i]]+=1) for i in range(len(test))]
for i in range(len(predictions)) :
    temp = torch.zeros(99)
    temp[predictions[i]]+=1
    ok.append(temp)

err = classifieur.erreur(y_test,torch.stack(ok))

print('erreur = ', err)
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
# classifieur.afficher_donnees_et_modele(X_train, y_train,X_test,y_test)