import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class TrainData:
    def __init__(self, filedata):
        self.leafClass = None
        self.data = None
        self.idLeaf = None
        self.filedata = filedata

        self.readData(self.filedata)

        self.image = None

    def readData(self, filedata):
        self.filedata = filedata
        # On recupere les caracteristiques des feuilles dans un dataFrame
        leaf_data = pd.read_csv(filedata, skiprows=0)

        self.leafClass = leaf_data["species"].values
        self.idLeaf = leaf_data["id"].values
        self.data = leaf_data.values[:, 2:]

    def read_image(self,
                   data_dir=r"leaf-classification\\images"):
        train_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()])

        tempList = []

        for id in range(1,len(self.idLeaf)):
            tempList.append((data_dir + "/" + str(id) + ".jpg", self.leafClass[id - 1]))

        train_data = MonDataset(tempList,
                                train_transforms)  # datasets.ImageFolder(data_dir, transform=train_transforms)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(self.idLeaf),
                                                   shuffle=True)  # , collate_fn=self.my_collate_fn)

        dataiter = torch.utils.data.DataLoader.__iter__((train_loader))
        images, labels = dataiter.__next__()

        print(type(images))
        print(images.shape)
        # print(labels.shape)

        self.images = images

        return train_loader

    def imshow(self):
        fig, ax = plt.subplots()

        image = self.images[0].numpy().transpose(1, 2, 0)
        print(image.shape)

        ax.imshow(image)
        ax.set_xticklabels('')
        ax.set_yticklabels('')

        plt.show()

        return ax

    def my_collate_fn(self, batch):
        images, labels = zip(*batch)
        return torch.stack(images), torch.tensor(labels)


class MonDataset(Dataset):
    def __init__(self, liste_tuples, transform=None):
        self.liste_tuples = liste_tuples
        self.transform = transform

    def __len__(self):
        return len(self.liste_tuples)

    def __getitem__(self, idx):
        image_path, classe = self.liste_tuples[idx]
        image = Image.open(image_path).convert('L')  # Assurez-vous que vos images sont en mode 'RGB'

        #encodage de la classe :
        Allclass = set([self.liste_tuples[i][1] for i in range(len(self.liste_tuples))])
        encoded_tensor = torch.zeros(len(Allclass))
        class_index = sorted(Allclass).index(classe)
        encoded_tensor[class_index] = 1

        # Appliquer des transformations si spécifiées
        if self.transform:
            image = self.transform(image)

        return image, encoded_tensor
