from sklearn.linear_model import Perceptron as SKlearnPerceptron
from .ClassifieurLineaire import StrategieClassification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA

class Perceptron(StrategieClassification):
    def __init__(self, learning_rate=0.01, max_iterations=1000,penalty='l2'):
        """
        Stratégie de classification utilisant le Perceptron de scikit-learn.

        :param learning_rate: Taux d'apprentissage pour le Perceptron.
        :param max_iterations: Nombre maximal d'itérations pour l'entraînement.
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.penalty=penalty
        self.perceptron_model = None

    def entrainer(self, x_train, t_train):
        """
        Entraîne le modèle de classification Perceptron de scikit-learn.

        :param x_train: Les données d'entraînement.
        :param t_train: Les étiquettes de classe cibles.
        """
        self.perceptron_model = SKlearnPerceptron(eta0=self.learning_rate, max_iter=self.max_iterations,penalty=self.penalty)
        self.perceptron_model.fit(x_train, t_train)
        self.w = self.perceptron_model.coef_
        self.w_0 = self.perceptron_model.intercept_

    def prediction(self, x):
        """
        Prédit la classe d'une nouvelle donnée x.

        :param classifieur: Une instance de ClassifieurLineaire.
        :param x: La donnée d'entrée à classifier.
        :return: 1 si la classe prédite est positive, -1 sinon.
        """
        if self.perceptron_model is not None:
            return self.perceptron_model.predict([x])[0]
        return 0  # Valeur par défaut si le modèle n'est pas encore entraîné

    def erreur(self, t, prediction):
        """
        Calcule l'erreur de classification.

        :param t: L'étiquette de classe réelle.
        :param prediction: La prédiction du modèle.
        :return: 1 si l'erreur est commise, 0 sinon.
        """
        return 1 if t != prediction else 0

    def parametres(self):
        """
        Retourne les paramètres du modèle
        """
        return self.w_0, self.w
    
    def afficher(self, x_train, t_train, x_test, t_test):
        le = LabelEncoder()

        # Encode training labels
        t_train_encoded = le.fit_transform(t_train)
        t_test_encoded = le.transform(t_test)

        h = 0.05  # Contrôle la résolution de la grille

        # Utilisez PCA pour réduire les dimensions à 2 pour la représentation
        pca = PCA(n_components=2)
        x_train_2d = pca.fit_transform(x_train)

        x_min, x_max = x_train_2d[:, 0].min() - .5, x_train_2d[:, 0].max() + .5
        y_min, y_max = x_train_2d[:, 1].min() - .5, x_train_2d[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Utilisez inverse_transform pour obtenir les coordonnées originales dans l'espace d'origine
        X_predict = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])

        # Classifier chaque point de la grille
        Z = self.perceptron_model.predict(X_predict)
        Z = le.transform(Z)
        Z = Z.reshape(xx.shape)

        
        plt.figure(figsize=(14, 8))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)  # Colorier les cases selon les prédictions

        plt.scatter(x_train[:, 0], x_train[:, 1], c=t_train_encoded, edgecolors='k', cmap=plt.cm.Paired)  # Tracer les données

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())

        plt.title('Frontières de décision')
        plt.show()
        
        return
        
        #visualizeClassificationAreas(self.perceptron_model,x_train, t_train_encoded,x_test, t_test_encoded)
        
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=(t_train_encoded * 100 + 20).astype(int), c=t_train_encoded)

        pente = -self.perceptron_model.coef_[0, 0] / self.perceptron_model.coef_[0, 1]
        xx_train = np.linspace(np.min(x_train[:, 0]) - 2, np.max(x_train[:, 0]) + 2, num=99)

        yy_train = pente * xx_train - self.perceptron_model.intercept_ / self.perceptron_model.coef_[0, 1]
        plt.plot(xx_train, yy_train)
        plt.title('Training data')

        # Encode testing labels
        t_test_encoded = le.transform(t_test)
        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=(t_test_encoded * 100 + 20).astype(int), c=t_test_encoded)

        pente = -self.perceptron_model.coef_[0, 0] / self.perceptron_model.coef_[0, 1]
        xx_test = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2, num=99)
        yy_test = pente * xx_test - self.perceptron_model.intercept_ / self.perceptron_model.coef_[0, 1]
        plt.plot(xx_test, yy_test)
        plt.title('Testing data')

        plt.show()

        return
        le = LabelEncoder()

        # Encode training labels
        t_train_encoded = le.fit_transform(t_train)
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=(t_train_encoded * 100 + 20).astype(int), c=t_train_encoded)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Training data')

        # Encode testing labels
        t_test_encoded = le.transform(t_test)
        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=(t_test_encoded * 100 + 20).astype(int), c=t_test_encoded)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Testing data')

        plt.show()
  
        
        return
        le = LabelEncoder()

        # Encode training labels
        t_train_encoded = le.fit_transform(t_train)
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=(t_train_encoded * 100 + 20), c=t_train_encoded)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Training data')

        # Encode testing labels
        t_test_encoded = le.transform(t_test)
        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=(t_test_encoded * 100 + 20), c=t_test_encoded)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Testing data')

        plt.show()
    
def visualizeClassificationAreas(usedClassifier,XtrainSet, ytrainSet,XtestSet, ytestSet, plotDensity=0.01 ):
    '''
    https://aleksandarhaber.com/solve-classification-problems-in-python-scikit-learn-and-visualize-the-classification-results-machine-learning-tutorial/
    This function visualizes the classification regions on the basis of the provided data

    usedClassifier      - used classifier

    XtrainSet,ytrainSet - train sets of features and target classes 
                    - the region colors correspond to to the class numbers in ytrainSet

    XtestSet,ytestSet   - test sets for visualizng the performance on test data that is not used for training
                    - this is optional argument provided if the test data is available

    plotDensity         - density of the area plot, smaller values are making denser and more precise plots

    IMPORTANT COMMENT:  -If the number of classes is larger than the number of elements in the list 
                    caller "colorList", you need to increase the number of colors in 
                    "colorList"
                    - The same comment applies to "markerClass" list defined below

    '''
    import numpy as np
    import matplotlib.pyplot as plt    
    # this function is used to create a "cmap" color object that is used to distinguish different classes
    from matplotlib.colors import ListedColormap 
    
    # this list is used to distinguish different classification regions
    # every color corresponds to a certain class number
    colorList=["blue", "green", "orange", "magenta", "purple", "red"]
    # this list of markers is used to distingush between different classe
    # every symbol corresponds to a certain class number
    markerClass=['x','o','v','#','*','>']
    
    # get the number of different classes 
    classesNumbers=np.unique(ytrainSet)
    numberOfDifferentClasses=classesNumbers.size
                
    # create a cmap object for plotting the decision areas
    cmapObject=ListedColormap(colorList, N=numberOfDifferentClasses)
                
    # get the limit values for the total plot
    x1featureMin=min(XtrainSet[:,0].min(),XtestSet[:,0].min())-0.5
    x1featureMax=max(XtrainSet[:,0].max(),XtestSet[:,0].max())+0.5
    x2featureMin=min(XtrainSet[:,1].min(),XtestSet[:,1].min())-0.5
    x2featureMax=max(XtrainSet[:,1].max(),XtestSet[:,1].max())+0.5
        
    # create the meshgrid data for the classifier
    x1meshGrid,x2meshGrid = np.meshgrid(np.arange(x1featureMin,x1featureMax,plotDensity),np.arange(x2featureMin,x2featureMax,plotDensity))
        
    # basically, we will determine the regions by creating artificial train data 
    # and we will call the classifier to determine the classes
    XregionClassifier = np.array([x1meshGrid.ravel(),x2meshGrid.ravel()]).T
    # call the classifier predict to get the classes
    predictedClassesRegion=usedClassifier.predict(XregionClassifier)
    # the previous code lines return the vector and we need the matrix to be able to plot
    predictedClassesRegion=predictedClassesRegion.reshape(x1meshGrid.shape)
    # here we plot the decision areas
    # there are two subplots - the left plot will plot the decision areas and training samples
    # the right plot will plot the decision areas and test samples
    fig, ax = plt.subplots(1,2,figsize=(15,8))
    ax[0].contourf(x1meshGrid,x2meshGrid,predictedClassesRegion,alpha=0.3,cmap=cmapObject)
    ax[1].contourf(x1meshGrid,x2meshGrid,predictedClassesRegion,alpha=0.3,cmap=cmapObject)
    
    # scatter plot of features belonging to the data set 
    for index1 in np.arange(numberOfDifferentClasses):
        ax[0].scatter(XtrainSet[ytrainSet==classesNumbers[index1],0],XtrainSet[ytrainSet==classesNumbers[index1],1], 
                alpha=1, c=colorList[index1], marker=markerClass[index1], 
                label="Class {}".format(classesNumbers[index1]), edgecolor='black', s=80)
        
        ax[1].scatter(XtestSet[ytestSet==classesNumbers[index1],0],XtestSet[ytestSet==classesNumbers[index1],1], 
                alpha=1, c=colorList[index1], marker=markerClass[index1],
                label="Class {}".format(classesNumbers[index1]), edgecolor='black', s=80)
        
    ax[0].set_xlabel("Feature X1",fontsize=14)
    ax[0].set_ylabel("Feature X2",fontsize=14)
    ax[0].text(0.05, 0.05, "Decision areas and training samples", transform=ax[0].transAxes, fontsize=14, verticalalignment='bottom')
    ax[0].legend(fontsize=14)
    
    ax[1].set_xlabel("Feature X1",fontsize=14)
    ax[1].set_ylabel("Feature X2",fontsize=14)
    ax[1].text(0.05, 0.05, "Decision areas and test samples", transform=ax[1].transAxes, fontsize=14, verticalalignment='bottom')
    #ax[1].legend()
    plt.savefig('classification_results.png')

        
"""
    def afficher(self, x_train, t_train, x_test=None, t_test=None, feature_names=None, class_names=None):
        if not feature_names:
            feature_names = [f'Feature {i}' for i in range(x_train.shape[1])]
        if not class_names:
            class_names = [f'Class {i}' for i in range(len(np.unique(t_train)))]

        # Create a mesh grid to plot decision boundaries
        h = .02  # step size in the mesh
        x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
        y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict the labels for each point in the mesh grid
        Z = self.perceptron_model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Reshape the predictions and plot the decision boundaries
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot the data points
        le = LabelEncoder()
        t_encoded = le.fit_transform(t_train)
        plt.scatter(x_train[:, 0], x_train[:, 1], c=t_encoded, edgecolors='k', cmap=plt.cm.Paired)

        # Labeling
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title('Perceptron Decision Boundaries')

        # Add a legend
        unique_classes = np.unique(t_train)
        class_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.Paired.colors[le.transform([cls])[0]],
                                markersize=10, label=class_names[le.transform([cls])[0]]) for cls in unique_classes]
        plt.legend(handles=class_handles)

        plt.show()
"""