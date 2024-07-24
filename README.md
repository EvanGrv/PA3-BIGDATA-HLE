
# Projet Annuel 2023/2024

## Classification des vaches, des moutons et des chèvres

### Auteurs:
- Lucas BONERE
- Evan GREVEN
- Hugo HOUNTONDJI

 Projet de Classification d'Images

Ce projet implémente une application de classification d'images, un script d'entraînement de modèle et des tests unitaires pour divers modèles de machine learning.

## Structure du Projet

### 📷_App.py
Application de classification d'images utilisant Streamlit. Cette application permet de télécharger une image, de la prétraiter et de prédire sa classe à l'aide d'un modèle de deep learning pré-entraîné.

### 📈_App_Train.py
Script d'entraînement de modèle utilisant Streamlit. Ce script charge les données, les prépare pour l'entraînement, entraîne un modèle de machine learning et affiche la précision sur les données de test.

### 🧪_Cas_De_Test.py
Tests unitaires pour vérifier le bon fonctionnement des modèles de machine learning. Ce fichier utilise la bibliothèque `unittest` pour tester les prédictions des modèles implémentés dans les fichiers `linear_model.py`, `mlp.py` et `rbf.py`.

### linear_model.py
Implémentation d'un modèle linéaire simplifié. Le modèle retourne simplement l'entrée comme prédiction.

### mlp.py
Implémentation d'un modèle MLP (Multilayer Perceptron). Ce fichier contient un modèle MLP avec une prédiction statique.

### rbf.py
Implémentation d'un modèle RBF (Radial Basis Function). Ce fichier contient un modèle RBF avec une prédiction statique.

## Fonctionnement du Projet

Le projet est structuré de manière à fournir une interface utilisateur pour la classification d'images ainsi qu'un script pour l'entraînement de modèles de machine learning. Les tests unitaires permettent de vérifier la validité des implémentations des modèles.

### Prérequis

- Python 3.7 ou plus
- Bibliothèques nécessaires (énumérées dans `requirements.txt`)

### Installation

1. Clonez le dépôt :
    ```bash
    git clone https://github.com/votre-utilisateur/votre-repo.git
    cd votre-repo
    ```

2. Créez un environnement virtuel et activez-le :
    ```bash
    python -m venv env
    source env/bin/activate  # Sur Windows, utilisez `env\Scripts\activate`
    ```

3. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

### Utilisation

#### 1. Application de Classification d'Images

Pour exécuter l'application de classification d'images :
```bash
streamlit run 📷_App.py
```
- Chargez une image en utilisant l'interface Streamlit.
- L'application affichera l'image téléchargée et prédira sa classe.

#### 2. Entraînement du Modèle

Pour exécuter le script d'entraînement du modèle :
```bash
streamlit run 📈_App_Train.py
```
- Le script chargera les données, entraînera un modèle et affichera la précision sur les données de test.

#### 3. Tests Unitaires

Pour exécuter les tests unitaires :
```bash
python -m unittest 🧪_Cas_De_Test.py
```
- Les tests vérifieront le bon fonctionnement des différentes implémentations de modèles.

## Modèles

### Modèle Linéaire
Implémenté dans `linear_model.py`.

### Modèle MLP (Multilayer Perceptron)
Implémenté dans `mlp.py`.

### Modèle RBF (Radial Basis Function)
Implémenté dans `rbf.py`.

## Contribution

Les contributions sont les bienvenues. Veuillez soumettre une pull request ou ouvrir une issue pour discuter de vos modifications.

## License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Détails Techniques et Explication du Code

import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome Train APP Model! 👋")

st.sidebar.success("Selectionne une page de l'application")

st.markdown(
    """
    
    -------------------------------------------------------------------
    
    ## Lien vers le [Github](https://github.com/EvanGrv/PA3-BIGDATA-HLE)
    
    -------------------------------------------------------------------
    
    # Projet Annuel 2023/2024

    ## Classification des vaches, des moutons et des chèvres
    
    ### Auteurs:
    - Lucas BONERE
    - Evan GREVEN
    - Hugo HOUNTONDJI
    
    ## Sommaire
    
    1. [Ressource Du Projet](#ressource-du-projet)
    2. [Introduction](#introduction)
    3. [Objectif Du Projet](#objectif-du-projet)
    4. [Mise en Place du Projet](#mise-en-place-du-projet)
    5. [Étape 1 : Construction d’un Data Set](#étape-1--construction-d’un-data-set)
    6. [Étape 2 : Modèle Linéaire](#étape-2--modèle-linéaire)
       - Importation de la Bibliothèque
       - Définition de la Structure LinearModel1
       - Fonction d'Entraînement
       - Initialisation des Paramètres
       - Boucle d'Entraînement
       - Calcul de la Sortie
       - Calcul des Erreurs
       - Mise à Jour des Poids et des Biais
       - Surveillance de la Convergence
       - Fonction de Calcul de la Perte
       - Fonction de Prédiction
       - Interfaçage avec le Code C
       - Fonction Principale
       - Tests Unitaires
    7. [PMC (Perceptron Multi Couche)](#pmc-perceptron-multi-couche)
       - Description du Code Rust
       - Description du Code Python
       - Cas de Test PMC
    
    ## Ressource Du Projet
    
    - [Lien Github](https://github.com/EvanGrv/PA3-BIGDATA-HLE.git)
    - Sites à scrapper :
      - [Vache sur Pexels](https://www.pexels.com/fr-fr/chercher/vache/)
      - [Chèvre sur Pexels](https://www.pexels.com/fr-fr/chercher/ch%C3%A8vre/)
      - [Vache sur Pixabay](https://pixabay.com/fr/images/search/vache/)
      - [Chèvre sur Pixabay](https://pixabay.com/fr/images/search/ch%c3%a8vre/)
    
    ## Introduction
    
    Le projet "Cow vs Goat" explore les capacités de l'apprentissage automatique pour distinguer des images de vaches, de chèvres et de moutons. Bien que cela puisse sembler simple, ce projet a des implications pour des systèmes de reconnaissance d'images plus avancés.
    
    ## Objectif Du Projet
    
    L'objectif est de tester quatre modèles d'apprentissage automatique différents :
    - **Modèle Linéaire (ML)**
    - **Perceptron Multi Couches (MLP)**
    - **Radial Basis Function Network (RBF)**
    - **Support Vector Machine (SVM)**
    
    Chaque modèle sera évalué sur sa capacité à distinguer entre vache, chèvre et mouton, en se basant sur des critères tels que la précision, le temps de calcul et la généralisation sur de nouvelles données.
    
    ## Mise en Place du Projet
    
    1. **Définition des objectifs**
    2. **Construction du Data Set** : Scrapping d'images à partir de sites internet pour créer une base de données.
    
    ## Étape 1 : Construction d’un Data Set
    
    ### Méthode de Scrapping
    - Utilisation initiale de Rust pour télécharger des images de Pixabay.
    - Adaptation pour utiliser Selenium en Python avec un webdriver Google Chrome pour automatiser la récupération des images.
    - Normalisation et nettoyage des données (redimensionnement et étiquetage).
    
    ## Étape 2 : Modèle Linéaire
    
    ### Implémentation
    1. **Importation de la Bibliothèque**
    2. **Définition de la Structure LinearModel1**
    3. **Fonction d'Entraînement** : Initialisation des paramètres, boucle d'entraînement, calcul des erreurs, mise à jour des poids et des biais.
    4. **Surveillance de la Convergence** : Calcul et affichage de la perte.
    5. **Interopérabilité avec le Code C** : Fonctions de prédiction et d'entraînement accessibles depuis du code C.
    6. **Tests Unitaires** : Validation du fonctionnement du modèle.
    
    ### Visualisation des Résultats
    - Tracé des points d'entraînement et de la ligne de séparation des classes à l'aide de matplotlib.
    
    ## PMC (Perceptron Multi Couche)
    
    ### Description du Code Rust
    - Création, entraînement et prédiction avec le modèle PMC.
    - Utilisation de la rétropropagation du gradient pour l'ajustement des poids.
    
    ### Description du Code Python
    - Utilisation de numpy, ctypes, et matplotlib pour interagir avec le modèle PMC compilé en Rust.
    - Définition des types de retour et des arguments des fonctions Rust.
    
    ### Cas de Test PMC
    - Tests spécifiques pour différents scénarios : Linear Simple, Linear Multiple, XOR, Multi Cross, Linear 3 Classe.
    
    ## Conclusion
    
    Ce projet a permis de comprendre comment combiner différents langages de programmation pour développer des modèles de classification efficaces. L'utilisation de Rust pour les calculs lourds et de Python pour l'interopérabilité et la visualisation a été particulièrement instructive.

"""
)
