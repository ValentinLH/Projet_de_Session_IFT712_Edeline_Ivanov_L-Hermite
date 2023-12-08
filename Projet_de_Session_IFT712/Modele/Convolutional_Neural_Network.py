
from Modele.ClassifieurLineaire import StrategieClassification

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim





class Net(nn.Module, StrategieClassification):
    def __init__(self):
        super().__init__()

        #L'explication des choix pour les valeur des

        # convolutional layer 1 & max pool layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU()#,
            #nn.MaxPool2d(kernel_size=2)
            )

        # convolutional layer 2 & max pool layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4),
            nn.MaxPool2d(kernel_size=2))

        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=3),
        #     nn.MaxPool2d(kernel_size=2))

        self.prelu = nn.PReLU()
        self.fc = nn.Linear(32 * 54 * 54, 99)
        self.fc = nn.Linear(387168, 99)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = 10
        
        self.batch_size = 16

        # Ajoutez des couches Dropout
        self.dropout = nn.Dropout(0.5)


        self.criterion = nn.CrossEntropyLoss()

        #self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9) #on parle sde paramettre du module
        self.optimizer = optim.Adam(self.parameters(), lr=0.01,betas=(0.9,0.99),weight_decay=1e-4) #on parle sde paramettre du module

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        #out = out.flatten()
        out = self.fc(out)
        out = self.dropout(out)
        return out

    def entrainer(self, x_train, t_train):
        inputs, labels = x_train, t_train

        # Utiliser DataLoader pour faciliter la gestion des mini-batchs
        train_dataset = torch.utils.data.TensorDataset(inputs, labels)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        
        for epoch in range(self.epochs):  # loop over the dataset multiple times


            running_loss = 0.0
            for i, (inputs_batch, labels_batch) in enumerate(train_loader):
       
            #for i in range(len(x_train)):
                # get the inputs; data is a list of [inputs, labels]
           

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs_batch)
                loss = self.criterion(outputs, labels_batch)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 0:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss /10:.5f}')
                    running_loss = 0.0

        print('Finished Training')

    def prediction(self, x):
        # This function should return the predicted class for input x
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs,1)
        return predicted.tolist()

    def parametres(self):
        # This function should return the parameters (weights and biases) of the model
        return list(self.parameters())

    def erreur(self, t, prediction):
        # This function should return the error between the true labels (t) and predicted labels (prediction)
        return self.criterion(prediction, t)

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
