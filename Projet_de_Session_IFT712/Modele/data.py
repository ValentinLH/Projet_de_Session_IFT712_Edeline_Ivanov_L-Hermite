import pandas as pd
class TrainData :
    def __init__(self,filedata):
        self.leafClass
        self.data
        self.idLeaf
        self.filedata = filedata

        self.readData(self.filedata)

    def readData(self,filedata):

        self.filedata = filedata
        # On recupere les caracteristiques des feuilles dans un dataFrame
        leaf_data = pd.read_csv(filedata, skiprows=0)

        self.leafClass = leaf_data["species"].values
        self.idLeaf = leaf_data["id"].values
        self.data = leaf_data.values[:,2:]
