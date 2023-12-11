import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, lr=0.001, epochs=15, batch_size=64, dropout=0.5):
        """
        Classe qui implemente la partie foncionnel du CNN

        :param lr: la valeur du pas d'apprentissage
        :param epochs: le nombre d'epochs realiser lors de l'entrainement
        :param batch_size: la taille du batch utiliser pour l'entrainement
        :param dropout: la valeur du DropOut

        """
        super().__init__()

        # Couche convolutive 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Couche convolutive 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(nn.Linear(32 * 54 * 54, 99),
                                nn.Softmax(dim=1)
                                )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = epochs

        self.batch_size = batch_size

        self.dropout = nn.Dropout(dropout)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.99),
                                    weight_decay=1e-4)  # on parle des paramettre du module

    def forward(self, x):
        """
        Fonction qui permet de realiser la propagation du CNN pour des donnees x
        :param x: donnees x
        :return: le resultats de la propagation avant sur x

        """
        out = self.layer1(x)
        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        out = self.dropout(out)

        return out

    def entrainer(self, x_train, t_train):

        """
        :param x_train: les donnees d'entrainement
        :param t_train: les etiquettes de classe liee au donnees d'entrainement
        """

        entre, classe = x_train, t_train

        # Utiliser DataLoader pour faciliter la gestion des mini-batchs
        donnees_entrainement = torch.utils.data.TensorDataset(entre, classe)
        chargeur_de_donnees = torch.utils.data.DataLoader(dataset=donnees_entrainement, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):

            perte_actuelle = 0.0
            for i, (inputs_batch, labels_batch) in enumerate(chargeur_de_donnees):

                self.optimizer.zero_grad()

                sortie = self.forward(inputs_batch)
                perte = self.criterion(sortie, labels_batch)
                perte.backward()
                self.optimizer.step()

                # Affichage de la perte
                perte_actuelle += perte.item()
                if i % 10 == 0:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {perte_actuelle / 20:.5f}')
                    perte_actuelle = 0.0

        print('Finished Training')

    def prediction(self, x):
        """
        :param x: les nouvelles donnees qu'on souhaite predire
        :return: les classes des donnees predites
        """
        with torch.no_grad():
            sortie = self.forward(x)
            _, predicted = torch.max(sortie, 1)
        return predicted.tolist()

    def erreur(self, t, prediction):
        """

        :param t: vrai etiquette de classe
        :param prediction: valeur predite par le CNN
        :return: return l'erreur calcule par la fonction d'erreur
        """
        return self.criterion(prediction, t)
