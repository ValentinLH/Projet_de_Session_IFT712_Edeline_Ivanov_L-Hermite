import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset


class TrainData:
    def __init__(self, fichier):
        self.leafClass = None
        self.data = None
        self.idLeaf = None
        self.fichier = fichier

        self.readData(self.fichier)

        self.image = None

    def readData(self, filedata):
        self.fichier = filedata

        # On recupere les caracteristiques des feuilles dans un dataFrame
        leaf_data = pd.read_csv(filedata, skiprows=0)

        self.leafClass = leaf_data["species"].values
        self.idLeaf = leaf_data["id"].values
        self.data = leaf_data.values[:, 2:]

    def read_image(self,
                   repertoire_images=r"leaf-classification\\images"):
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])

        ListeTemporaire = []

        for id in range(1, len(self.idLeaf)):
            ListeTemporaire.append((repertoire_images + "/" + str(id) + ".jpg", self.leafClass[id - 1]))

        donnees_entrainement = MonDataset(ListeTemporaire,
                                          train_transforms)

        chargeur_de_donnees = torch.utils.data.DataLoader(donnees_entrainement, batch_size=len(self.idLeaf),
                                                          shuffle=True)

        dataiter = torch.utils.data.DataLoader.__iter__((chargeur_de_donnees))
        images, classes = dataiter.__next__()

        self.image = images

        return chargeur_de_donnees

    def imshow(self):
        fig, ax = plt.subplots()

        image = self.image[0].numpy().transpose(1, 2, 0)
        print(image.shape)

        ax.imshow(image)
        ax.set_xticklabels('')
        ax.set_yticklabels('')

        plt.show()

        return ax


class MonDataset(Dataset):
    def __init__(self, liste_tuples, transform=None):
        self.liste_tuples = liste_tuples
        self.transform = transform

    def __len__(self):
        return len(self.liste_tuples)

    def __getitem__(self, idx):
        chemin_image, classe = self.liste_tuples[idx]
        image = Image.open(chemin_image).convert('L')

        # encodage de la classe :
        TouteLesClasses = set([self.liste_tuples[i][1] for i in range(len(self.liste_tuples))])
        tensor_encode = torch.zeros(len(TouteLesClasses))
        classe_index = sorted(TouteLesClasses).index(classe)
        tensor_encode[classe_index] = 1

        # Appliquer des transformations si spécifiées
        if self.transform:
            image = self.transform(image)

        return image, tensor_encode
