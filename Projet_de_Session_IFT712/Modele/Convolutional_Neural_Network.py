
from Modele.ClassifieurLineaire import StrategieClassification

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



"""
####################################################################

Net est une classe hybride, il y avais une récursion self.model = Net()
car tout les Net() appellait dans leur constructeur Net()

ensuite
on a plein de maniere de l'implementé

on peut faire un modele "complet" CNN ne marchant qu'avec torch et 
notre classe encapsulerais celle ci avec la structure de la stratégie et des sanity checks

ou on peut faire un module incestueux combinant les deux, nous allons en reparler dans un futur proche

Bisous





####################################################################
"""


class Net(nn.Module, StrategieClassification):
    def __init__(self):
        super().__init__()

        #L'explication des choix pour les valeur des

        # convolutional layer 1 & max pool layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.MaxPool2d(kernel_size=2))

        # convolutional layer 2 & max pool layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4),
            nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Linear(32 * 54 * 54, 6)

        #self.model = Net()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = 2

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9) #on parle sde paramettre du module

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #out = out.reshape(out.size(0), -1)
        out = out.flatten()
        out = self.fc(out)
        return out

    def entrainer(self, x_train, t_train):
        for epoch in range(self.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i in range(len(x_train)):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = x_train[i], t_train[i]

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')

    def prediction(self, x):
        pass

    
    def parametres(self):
        pass
    

    def erreur(self, t, prediction):
        pass

    def afficher(self, x_train, t_train, x_test, t_test):
        pass





# class Convolutional_Neural_Network(StrategieClassification) :
#     def __init__(self):
#         self.CNN_Model = None
#
#
#     def entrainer(self, x_train, t_train):
#         return
#
#     def prediction(self, x):
#         return
#
#     def parametres(self):
#         return
#
#     def erreur(self,t, prediction):
#         return
#
#     def afficher(self, x_train, t_train, x_test, t_test):
#         return