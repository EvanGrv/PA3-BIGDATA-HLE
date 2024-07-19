import os
import sys
import re
import streamlit as st
from PIL import Image
import numpy as np

sys.path.append('C:\\Users\\lucho\\Documents\\cours\\projet_annuel\\rbf')
import rbf

sys.path.append('C:\\Users\\lucho\\Documents\\cours\\projet_annuel\\linear_model')
import linear_model

sys.path.append('C:\\Users\\lucho\\Documents\\cours\\projet_annuel\\mlp')
import mlp

st.set_page_config(
    page_title="App",
    page_icon="📷",
)


def save_uploaded_file(uploadedfile, folder="uploaded_images"):
    # Créez un répertoire temporaire s'il n'existe pas
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Sauvegardez le fichier dans ce répertoire
    file_path = os.path.join(folder, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return os.path.abspath(file_path)


def extract_array_from_filename(filename):
    # Utiliser une expression régulière pour extraire la partie entre crochets
    match = re.search(r'\[(.*?)\]', filename)
    if match:
        # Extraire la chaîne trouvée par l'expression régulière
        array_str = match.group(1)
        # Convertir la chaîne en liste d'entiers
        array = list(map(int, array_str.split(',')))
        return array
    else:
        raise ValueError("Le format du fichier ne contient pas de tableau entre crochets")


def predict(json, model, image_path):
    if model == "MLP":
        mlp_model = mlp.lib.load_mlp(json.encode('utf-8'))
        layers = extract_array_from_filename(json)
        npl = np.array(layers)

        class_names = ['vache', 'chevre', 'mouton']

        predicted_class, predicted_output = mlp.predict_image_class(mlp_model, image_path, npl, class_names)

        print(f"{predicted_class} {predicted_output}")


st.title("Image Uploader 📷")

# Interface pour uploader une image
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

image_path = ""

if uploaded_file is not None:
    # Lire l'image
    image = Image.open(uploaded_file)

    # Afficher l'image
    st.image(image, caption='Image chargée', use_column_width=True)

    # Afficher le chemin de l'image
    image_path = save_uploaded_file(uploaded_file)
    st.write(f"Chemin absolu de l'image : {image_path}")

model = st.selectbox(
    "Sélectionner un model",
    ("Linear Model", "MLP", "RBF"),
    index=None,
    placeholder="Select model...",
)

st.write("You selected:", model)

st.write("Charger un model pré-entrainé")

default_json_path = "C:\\Users\\lucho\\Documents\\cours\\projet_annuel\\app\\save_model\\model_mlp_[4800, 10, 10, 3]_10_0.001.json"

# Interface pour uploader un fichier
json_file = st.text_input("Entrez le chemin vers le model pré-entrainé", value=default_json_path)

st.button("Start", key="verif", on_click=predict(json_file, model, image_path), type="secondary")
