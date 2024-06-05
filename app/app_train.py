import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.append('C:\\Users\\lbone\\Documents\\Project\\Python_test\\rbf')

import rbf


def rbf_test_show(xx, yy, grid_predictions, X_train, Y_train):
    # Tracer les points d'entraînement
    plt.scatter(X_train[:, 0], X_train[:, 1], c=np.argmax(Y_train, axis=1))

    k = len(Y_train[0])
    if k <= 2:
        # Tracer la séparation des classes
        contour = grid_predictions[:, 0].reshape(xx.shape)
        plt.contourf(xx, yy, contour, levels=[-np.inf, 0, np.inf], colors=['blue', 'red'], alpha=0.5)

    else:
        # Tracer la séparation des classes
        contour = np.argmax(grid_predictions, axis=1).reshape(xx.shape)
        plt.contourf(xx, yy, contour, levels=[-np.inf, 0.5, 1.5, np.inf], colors=['blue', 'red', 'green'],
                     alpha=0.4)

    st.pyplot(plt.gcf())

    plt.clf()


def verification(model, train, test):
    match model:
        case "Linear Model":
            if train == "True":
                st.write("True")
            else:
                st.write("LM")
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
                        xx, yy, grid_predictions, X_train, Y_train = rbf.linear_simple()
                        rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

                    case "Linear Multiple":
                        xx, yy, grid_predictions, X_train, Y_train = rbf.linear_multiple()
                        rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

                    case "Xor":
                        xx, yy, grid_predictions, X_train, Y_train = rbf.xor()
                        rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

                    case "Cross":
                        xx, yy, grid_predictions, X_train, Y_train = rbf.cross()
                        rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

                    case "Multi Linear 3 classes":
                        xx, yy, grid_predictions, X_train, Y_train = rbf.multi_linear_classes()
                        rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

                    case "Multi Cross":
                        xx, yy, grid_predictions, X_train, Y_train = rbf.multi_cross()
                        rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

                    case "All":
                        xx, yy, grid_predictions, X_train, Y_train = rbf.linear_simple()
                        rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

                        xx, yy, grid_predictions, X_train, Y_train = rbf.linear_multiple()
                        rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

                        xx, yy, grid_predictions, X_train, Y_train = rbf.xor()
                        rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

                        xx, yy, grid_predictions, X_train, Y_train = rbf.cross()
                        rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

                        xx, yy, grid_predictions, X_train, Y_train = rbf.multi_linear_classes()
                        rbf_test_show(xx, yy, grid_predictions, X_train, Y_train)

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
