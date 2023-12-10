# Projet_de_Session_IFT712_Edeline_Ivanov_L'Hermite
 
Ce projet regroupe le code écrit par L'HERMITE Valentin, EDELINE Maxime et IVANOV Nicolas dans le cadre du projet de fin de session de IFT712 - Techniques d'apprentissage.

## Présentation

Nous allons travailler sur la base de données "Leaf Classification", fournie par Kaggle. On possède une base de données composé d'un ensemble de feuilles. Chaque feuille est caractérisé par 192 attributs. Le modèle doit être capable de déterminer l'espèce d'appartenance de la feuille parmi 99 espèces. Ainsi, l'objectif est d'entraîner le modèle pour qu'il détermine la classe d'appartenance d'une feuilles qu'on lui fourni parmi 99 classes.
Pour cette exercice, nous avons implémenté ces algorithmes:
- ADA Boost
- Perceptron
- Perceptron multicouches
- Random Forest
- Réseaux de neurone convolutifs
- Machines à vecteurs de support

## Structure des fichiers

Le fichier principal est projet_IFT712.ipynb, il comprends les informations nécessaires à l'exécution de nos classifieurs
Dans le dossier Modele, on retrouve tous les fichiers d'implémentation de nos algorithmes et un dossier RechercheHyperparameter contenant les fichiers d'implémentation de nos algorithmes de recherche d'hyperparamètres