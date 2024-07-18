import os

import streamlit as st
from PIL import Image

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


st.title("Image Uploader 📷")

# Interface pour uploader une image
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lire l'image
    image = Image.open(uploaded_file)

    # Afficher l'image
    st.image(image, caption='Image chargée', use_column_width=True)

    # Afficher le chemin de l'image
    image_path = save_uploaded_file(uploaded_file)
    st.write(f"Chemin absolu de l'image : {image_path}")
