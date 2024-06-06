import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.append('C:\\Users\\lbone\\Documents\\Project\\Python_test\\PA3-BIGDATA-HLE\\rbf')
import rbf

sys.path.append('C:\\Users\\lbone\\Documents\\Project\\Python_test\\PA3-BIGDATA-HLE\\linear_model')
import linear_model

# sys.path.append('C:\\Users\\lbone\\Documents\\Project\\Python_test\\PA3-BIGDATA-HLE\\mlp')
# import mlp


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


def verification(model, train, test):
    match model:
        case "Linear Model":
            if train == "True":
                st.write("True")
            else:
                if train == "True":
                    st.write("True")
                elif train == "False":
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
                            #xx, yy, grid_predictions, X_train, Y_train = linear_model.xor()
                            #rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

                        case "Cross":
                            st.write("""
                            ## Cross
                            """)
                            #xx, yy, grid_predictions, X_train, Y_train = linear_model.cross()
                            #rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

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

                        case _:
                            st.write("Selectionner un test")
                else:
                    st.write("Selectionner un entrainement")
        case "MLP":
            if train == "True":
                st.write("True")
            else:
                st.write("MLP")
        case "RBF":
            if train == "True":
                st.write("True")
            elif train == "False":
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

if train == "False":
    test = st.selectbox(
        "Choisi un test pour le model",
        ("Linear Simple", "Linear Multiple", "Xor", "Cross", "Multi Linear 3 classes", "Multi Cross", "All"),
        index=None,
        placeholder="Select Test...",
    )

    st.write("You selected:", test)

st.button("Start", key="verif", on_click=verification(model, train, test), type="secondary")
