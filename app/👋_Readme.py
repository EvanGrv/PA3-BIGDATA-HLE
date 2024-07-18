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