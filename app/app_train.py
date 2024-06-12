import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.append('..\\rbf')
import rbf

sys.path.append('..\\linear_model')
import linear_model

sys.path.append('..\\mlp')
import mlp


def rbf_test_show(xx, yy, grid_predictions, X_train, Y_train):
    k = len(Y_train[0])
    if k <= 2:
        # Tracé des points d'entraînement avec des couleurs différentes pour chaque classe
        class_0 = X_train[Y_train[:, 0] < 0]
        class_1 = X_train[Y_train[:, 0] > 0]

        plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', edgecolor='k', label='Classe 0')
        plt.scatter(class_1[:, 0], class_1[:, 1], color='red', edgecolor='k', label='Classe 1')

        # Tracer la séparation des classes
        contour = grid_predictions[:, 0].reshape(xx.shape)
        plt.contourf(xx, yy, contour, levels=[-np.inf, 0, np.inf], colors=['blue', 'red'], alpha=0.5)

    else:
        # Tracé des points d'entraînement avec des couleurs différentes pour chaque classe
        class_0 = X_train[Y_train[:, 0] < 0]
        class_1 = X_train[Y_train[:, 0] > 0]
        class_2 = X_train[np.argmax(Y_train, axis=1) == 2]

        plt.scatter(class_0[:, 0], class_0[:, 1], color='red', edgecolor='k', label='Classe 0')
        plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', edgecolor='k', label='Classe 1')
        plt.scatter(class_2[:, 0], class_2[:, 1], color='green', edgecolor='k', label='Classe 2')

        # Tracer la séparation des classes
        contour = np.argmax(grid_predictions, axis=1).reshape(xx.shape)
        plt.contourf(xx, yy, contour, levels=[-np.inf, 0.5, 1.5, np.inf], colors=['blue', 'red', 'green'], alpha=0.4)

    st.pyplot(plt.gcf())

    plt.clf()

def regression_test_show(inputs, outputs):
    num_feature = inputs.shape[1]
    if num_feature == 1:
        # Régression linéaire avec pseudo-inverse de Moore Penrose pour une caractéristique
        ones = np.ones((inputs.shape[0], 1))
        X = np.hstack([ones, inputs])  # Ajout du biais
        Y = outputs

        # Calcul de la pseudo-inverse et des weights
        X_prime = np.linalg.pinv(X)
        w = X_prime.dot(Y)

        # Traçons les résultats en 2D
        plt.scatter(inputs, outputs, color='blue', label='Data points')
        x_vals = np.linspace(inputs.min(), inputs.max(), 100)
        y_vals = w[0] + w[1] * x_vals

        plt.plot(x_vals, y_vals, color='red', label='Regression line')
        plt.xlabel('Feature')
        plt.ylabel('Output')
        plt.legend()

        st.pyplot(plt.gcf())

        plt.clf()


    elif num_feature == 2:
        # Régression linéaire avec pseudo-inverse de Moore Penrose pour deux caractéristiques
        ones = np.ones((inputs.shape[0], 1))
        X = np.hstack([ones, inputs])
        Y = outputs

        # Calcul de la pseudo-inverse et des poids
        X_prime = np.linalg.pinv(X)
        w = X_prime.dot(Y)

        # Tracé des résultats en 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(inputs[:, 0], inputs[:, 1], outputs, color='blue', label='Data points')

        x_surf, y_surf = np.meshgrid(np.linspace(inputs[:, 0].min(), inputs[:, 0].max(), 100),
                                     np.linspace(inputs[:, 1].min(), inputs[:, 1].max(), 100))
        z_surf = w[0] + w[1] * x_surf + w[2] * y_surf
        ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5, label='Regression plane')

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Output')

        plt.legend()
        st.pyplot(plt.gcf())

        plt.clf()


def linear_model_test(test):
    match test:
        case "Linear Simple":
            st.write("""
            ## Linear Simple
            """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.linear_simple()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "Linear Multiple":
            st.write("""
            ## Linear Multiple
            """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.linear_multiple()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "Xor":
            st.write("""
            ## XOR
            """)
            # xx, yy, grid_predictions, X_train, Y_train = linear_model.xor()
            # rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "Cross":
            st.write("""
            ## Cross
            """)
            # xx, yy, grid_predictions, X_train, Y_train = linear_model.cross()
            # rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "Multi Linear 3 classes":
            st.write("""
            ## Multi Linear 3 classes
            """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.multi_linear_classes()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "Multi Cross":
            st.write("""
            ## Multi Cross
            """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.multi_cross()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "Linear Simle 2D":
            st.write("""
            ## Linear Simle 2D
            """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.linear_simple_2D()
            regression_test_show(X_train, Y_train)
        case "Non Linear Simle 2D":
            st.write("""
                ## Non Linear Simle 2D
                """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.non_linear_simple_2D()
            regression_test_show(X_train, Y_train)

        case "Linear Simple 3D":
            st.write("""
                    ## Linear Simple 3D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.linear_simple_3D()
            regression_test_show(X_train, Y_train)

        case "Linear Tricky 3D":
            st.write("""
                    ## Linear Tricky 3D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.linear_tricky_3D()
            regression_test_show(X_train, Y_train)

        case "Non Linear Tricky 3D":
            st.write("""
                    ## Linear Tricky 3D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.non_linear_tricky_3D()
            regression_test_show(X_train, Y_train)

        case "All-cla":
            st.write("""
            ## Linear Simple
            """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.linear_simple()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Linear Multiple
            """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.linear_multiple()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## XOR
            """)

            st.write("""
            ### XOR ne fonctionne pas
            """)
            # xx, yy, grid_predictions, X_train, Y_train = linear_model.xor()
            # rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Cross
            """)

            st.write("""
            ### Cross ne fonctionne pas
            """)
            # xx, yy, grid_predictions, X_train, Y_train = linear_model.cross()
            # rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Multi Linear 3 classes
            """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.multi_linear_classes()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Multi Cross
            """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.multi_cross()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "All-reg":
            st.write("""
                    ## Linear Simle 2D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.linear_simple_2D()
            regression_test_show(X_train, Y_train)

            st.write("""
                    ## Non Linear Simle 2D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.non_linear_simple_2D()
            regression_test_show(X_train, Y_train)

            st.write("""
                    ## Linear Simple 3D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.linear_simple_3D()
            regression_test_show(X_train, Y_train)

            st.write("""
                    ## Linear Tricky 3D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.linear_tricky_3D()
            regression_test_show(X_train, Y_train)

            st.write("""
                    ## Linear Tricky 3D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.non_linear_tricky_3D()
            regression_test_show(X_train, Y_train)

        case "All":
            st.write("""
            ## Linear Simple
            """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.linear_simple()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Linear Multiple
            """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.linear_multiple()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## XOR
            """)

            st.write("""
            ### XOR ne fonctionne pas
            """)
            # xx, yy, grid_predictions, X_train, Y_train = linear_model.xor()
            # rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Cross
            """)

            st.write("""
            ### Cross ne fonctionne pas
            """)
            # xx, yy, grid_predictions, X_train, Y_train = linear_model.cross()
            # rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Multi Linear 3 classes
            """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.multi_linear_classes()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Multi Cross
            """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.multi_cross()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)


            st.write("""
                    ## Linear Simle 2D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.linear_simple_2D()
            regression_test_show(X_train, Y_train)

            st.write("""
                    ## Non Linear Simle 2D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.non_linear_simple_2D()
            regression_test_show(X_train, Y_train)

            st.write("""
                    ## Linear Simple 3D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.linear_simple_3D()
            regression_test_show(X_train, Y_train)

            st.write("""
                    ## Linear Tricky 3D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.linear_tricky_3D()
            regression_test_show(X_train, Y_train)

            st.write("""
                    ## Linear Tricky 3D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = linear_model.non_linear_tricky_3D()
            regression_test_show(X_train, Y_train)

        case _:
            st.write("Selectionner un test")


def rbf_model_test(test):
    match test:
        case "Linear Simple":
            st.write("""
            ## Linear Simple
            """)
            xx, yy, grid_predictions, X_train, Y_train = rbf.linear_simple()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "Linear Multiple":
            st.write("""
            ## Linear Multiple
            """)
            xx, yy, grid_predictions, X_train, Y_train = rbf.linear_multiple()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "Xor":
            st.write("""
            ## XOR
            """)
            xx, yy, grid_predictions, X_train, Y_train = rbf.xor()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "Cross":
            st.write("""
            ## Cross
            """)
            xx, yy, grid_predictions, X_train, Y_train = rbf.cross()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "Multi Linear 3 classes":
            st.write("""
            ## Multi Linear 3 classes
            """)
            xx, yy, grid_predictions, X_train, Y_train = rbf.multi_linear_classes()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "Multi Cross":
            st.write("""
            ## Multi Cross
            """)
            xx, yy, grid_predictions, X_train, Y_train = rbf.multi_cross()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "All":

            st.write("""
            ## Linear Simple
            """)

            xx, yy, grid_predictions, X_train, Y_train = rbf.linear_simple()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Linear Multiple
            """)

            xx, yy, grid_predictions, X_train, Y_train = rbf.linear_multiple()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## XOR
            """)

            xx, yy, grid_predictions, X_train, Y_train = rbf.xor()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Cross
            """)

            xx, yy, grid_predictions, X_train, Y_train = rbf.cross()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Multi Linear 3 classes
            """)

            xx, yy, grid_predictions, X_train, Y_train = rbf.multi_linear_classes()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Multi Cross
            """)

            xx, yy, grid_predictions, X_train, Y_train = rbf.multi_cross()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case _:
            st.write("Selectionner un test")


def mlp_model_test(test):
    match test:
        case "Linear Simple":
            st.write("""
            ## Linear Simple
            """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.linear_simple()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "Linear Multiple":
            st.write("""
            ## Linear Multiple
            """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.linear_multiple()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "Xor":
            st.write("""
            ## XOR
            """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.xor()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "Cross":
            st.write("""
            ## Cross
            """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.cross()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "Multi Linear 3 classes":
            st.write("""
            ## Multi Linear 3 classes
            """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.multi_linear_classes()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "Multi Cross":
            st.write("""
            ## Multi Cross
            """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.multi_cross()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "Linear Simle 2D":
            st.write("""
                    ## Linear Simle 2D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.linear_simple_2D()
            regression_test_show(X_train, Y_train)
        case "Non Linear Simle 2D":
            st.write("""
                        ## Non Linear Simle 2D
                        """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.non_linear_simple_2D()
            regression_test_show(X_train, Y_train)

        case "Linear Simple 3D":
            st.write("""
                            ## Linear Simple 3D
                            """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.linear_simple_3D()
            regression_test_show(X_train, Y_train)

        case "Linear Tricky 3D":
            st.write("""
                            ## Linear Tricky 3D
                            """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.linear_tricky_3D()
            regression_test_show(X_train, Y_train)

        case "Non Linear Tricky 3D":
            st.write("""
                            ## Linear Tricky 3D
                            """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.non_linear_tricky_3D()
            regression_test_show(X_train, Y_train)

        case "All-cla":

            st.write("""
            ## Linear Simple
            """)

            xx, yy, grid_predictions, X_train, Y_train = mlp.linear_simple()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Linear Multiple
            """)

            xx, yy, grid_predictions, X_train, Y_train = mlp.linear_multiple()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## XOR
            """)

            xx, yy, grid_predictions, X_train, Y_train = mlp.xor()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Cross
            """)

            xx, yy, grid_predictions, X_train, Y_train = mlp.cross()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Multi Linear 3 classes
            """)

            xx, yy, grid_predictions, X_train, Y_train = mlp.multi_linear_classes()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Multi Cross
            """)

            xx, yy, grid_predictions, X_train, Y_train = mlp.multi_cross()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

        case "All-reg":
            st.write("""
                    ## Linear Simle 2D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.linear_simple_2D()
            regression_test_show(X_train, Y_train)

            st.write("""
                    ## Non Linear Simle 2D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.non_linear_simple_2D()
            regression_test_show(X_train, Y_train)

            st.write("""
                    ## Linear Simple 3D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.linear_simple_3D()
            regression_test_show(X_train, Y_train)

            st.write("""
                    ## Linear Tricky 3D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.linear_tricky_3D()
            regression_test_show(X_train, Y_train)

            st.write("""
                    ## Linear Tricky 3D
                    """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.non_linear_tricky_3D()
            regression_test_show(X_train, Y_train)

        case "All":

            st.write("""
            ## Linear Simple
            """)

            xx, yy, grid_predictions, X_train, Y_train = mlp.linear_simple()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Linear Multiple
            """)

            xx, yy, grid_predictions, X_train, Y_train = mlp.linear_multiple()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## XOR
            """)

            xx, yy, grid_predictions, X_train, Y_train = mlp.xor()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Cross
            """)

            xx, yy, grid_predictions, X_train, Y_train = mlp.cross()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Multi Linear 3 classes
            """)

            xx, yy, grid_predictions, X_train, Y_train = mlp.multi_linear_classes()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
            ## Multi Cross
            """)

            xx, yy, grid_predictions, X_train, Y_train = mlp.multi_cross()
            rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

            st.write("""
                        ## Linear Simle 2D
                        """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.linear_simple_2D()
            regression_test_show(X_train, Y_train)

            st.write("""
                        ## Non Linear Simle 2D
                        """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.non_linear_simple_2D()
            regression_test_show(X_train, Y_train)

            st.write("""
                        ## Linear Simple 3D
                        """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.linear_simple_3D()
            regression_test_show(X_train, Y_train)

            st.write("""
                        ## Linear Tricky 3D
                        """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.linear_tricky_3D()
            regression_test_show(X_train, Y_train)

            st.write("""
                        ## Linear Tricky 3D
                        """)
            xx, yy, grid_predictions, X_train, Y_train = mlp.non_linear_tricky_3D()
            regression_test_show(X_train, Y_train)

        case _:
            st.write("Selectionner un test")


def verification(model, train, test, load):
    match model:
        case "Linear Model":
            if train == "True":
                st.write("True")
            elif train == "False":
                if load == "False":
                    linear_model_test(test)
                # elif load == "True":
                else:
                    st.write("Selectionner load")
            else:
                st.write("Selectionner un entrainement")
        case "MLP":
            if train == "True":
                st.write("True")
            elif train == "False":
                if load == "False":
                    mlp_model_test(test)
                    print()
                # elif load == "True":
                else:
                    st.write("Selectionner load")

            else:
                st.write("Selectionner un entrainement")
        case "RBF":
            if train == "True":
                st.write("True")
            elif train == "False":
                if load == "False":
                    rbf_model_test(test)
                # elif load == "True":
                else:
                    st.write("Selectionner load")

            else:
                st.write("Selectionner un entrainement")
        case _:
            st.write("Selectionner un model")


st.write("""
# Train App
""")

model = st.selectbox(
    "Sélectionner un model",
    ("Linear Model", "MLP", "RBF"),
    index=None,
    placeholder="Select model...",
)

st.write("You selected:", model)

train = st.selectbox(
    "Veux-tu lancer un entrainement",
    ("True", "False"),
    index=None,
    placeholder="Select Train...",
)

st.write("You selected:", train)

test = "new"
load = "new"

if train == "False":

    load = st.selectbox(
        "Veux tu charger un model pré-entrainé ?",
        ("True", "False"),
        index=None,
        placeholder="Select Load...",
    )

    st.write("You selected:", load)

    if load == "True":
        st.write("Charger un model pré-entrainé")

        default_json_path = "../"

        # Interface pour uploader un fichier
        json_file = st.text_input("Entrez le chemin vers le model pré-entrainé", value=default_json_path)

    elif load == "False":
        test = st.selectbox(
            "Choisi un test pour le model",
            ("Linear Simple", "Linear Multiple", "Xor", "Cross", "Multi Linear 3 classes", "Multi Cross",
             "Linear Simle 2D", "Non Linear Simle 2D", "Linear Simple 3D", "Linear Tricky 3D", "Non Linear Tricky 3D",
             "All-cla", "All-reg", "All"),
            index=None,
            placeholder="Select Test...",
        )

        st.write("You selected:", test)

elif train == "True":
    st.write("Charger votre DataSet")

    default_data_path = "../DataSet/"

    data_path = st.text_input("Entrez le chemin vers le DataSet", value=default_data_path)

st.button("Start", key="verif", on_click=verification(model, train, test, load), type="secondary")
