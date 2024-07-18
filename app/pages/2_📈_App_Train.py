import os
from streamlit_autorefresh import st_autorefresh
import streamlit as st
import subprocess
import sys
import shlex

sys.path.append('C:\\Users\\lucho\\Documents\\cours\\projet_annuel\\rbf')
import rbf

sys.path.append('C:\\Users\\lucho\\Documents\\cours\\projet_annuel\\linear_model')
import linear_model

sys.path.append('C:\\Users\\lucho\\Documents\\cours\\projet_annuel\\mlp')
import mlp

st.set_page_config(
    page_title="Train App",
    page_icon="📈",
)

LOG_FILE = 'train_log.txt'


# Fonction pour convertir une chaîne de caractères en une liste d'entiers
def parse_int_list(input_str):
    try:
        return [int(i) for i in input_str.strip('[]').split(',')]
    except ValueError:
        st.error("Veuillez entrer une liste d'entiers valide.")
        return []


# Fonction pour exécuter un entraînement et capturer les sorties en temps réel
def run_training_command(command):
    # Crée le fichier de logs s'il n'existe pas
    if not os.path.exists(LOG_FILE):
        open(LOG_FILE, 'w').close()

    with open(LOG_FILE, 'w') as logs_file:
        process = subprocess.Popen(shlex.split(command), stdout=logs_file, stderr=logs_file, text=True)
        process.wait()
    return process.returncode


# Fonction pour nettoyer le fichier de log
def clean_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w') as log_file:
            log_file.write("")


def verification(model, train, class1, class2, class3, iteration_count, alpha, n=None):
    try:
        match model:
            case "Linear Model":
                if train == "True":
                    st.write("True")
                elif train == "False":
                    st.write("False")
                else:
                    st.write("Selectionner un entrainement")
            case "MLP":
                if train == "True":
                    # mlp.test_train_image(n, iteration_count, alpha, str(class1), str(class2), str(class3))
                    command = f'python -c "import sys; sys.path.append(r\'C:\\Users\\lucho\\Documents\\cours\\projet_annuel\\mlp\'); import mlp; mlp.test_train_image({n}, {iteration_count}, {alpha}, \'{class1}\', \'{class2}\', \'{class3}\')"'
                    run_training_command(command)
                elif train == "False":
                    st.write("False")
                else:
                    st.write("Selectionner un entrainement")
            case "RBF":
                if train == "True":
                    st.write("True")
                elif train == "False":
                    st.write("False")
                else:
                    st.write("Selectionner un entrainement")
            case _:
                st.write("Selectionner un model")
    except Exception as e:
        st.error(f"Erreur : {e}")


st.write("""
# Train App 📈
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

if train == "False":
    st.write("Charger un model pré-entrainé")

    default_json_path = "../../"

    # Interface pour uploader un fichier
    json_file = st.text_input("Entrez le chemin vers le model pré-entrainé", value=default_json_path)


elif train == "True":
    st.write("Charger votre DataSet")

    st.write("Classe 1 :")

    default_data_path1 = "../DataSet/vache"

    data_path1 = st.text_input("Entrez le chemin vers le DataSet", value=default_data_path1)

    st.write("Classe 2 :")

    default_data_path2 = "../DataSet/chevre"

    data_path2 = st.text_input("Entrez le chemin vers le DataSet", value=default_data_path2)

    st.write("Classe 3 :")

    default_data_path3 = "../DataSet/mouton"

    data_path3 = st.text_input("Entrez le chemin vers le DataSet", value=default_data_path3)

    # Champs de texte pour saisir les paramètres du modèle
    n_input = st.text_input("Entrez la liste des paramètres n (par ex. [4800, 512, 256, 3])", "[4800, 512, 256, 3]")
    iteration_count_input = st.text_input("Entrez le nombre d'itérations", "1000")
    alpha_input = st.text_input("Entrez la valeur de alpha", "0.001")

    # Convertir les entrées utilisateur en types appropriés
    n = parse_int_list(n_input)
    try:
        iteration_count = int(iteration_count_input)
    except ValueError:
        st.error("Veuillez entrer un entier valide pour iteration_count.")
        iteration_count = None

    try:
        alpha = float(alpha_input)
    except ValueError:
        st.error("Veuillez entrer une valeur flottante valide pour alpha.")
        alpha = None

    # Bouton pour lancer la vérification
    if st.button("Start", key="verif", type="secondary"):
        if n and iteration_count is not None and alpha is not None:
            verification(model, train, data_path1, data_path2, data_path3, iteration_count, alpha, n)
        else:
            st.error("Veuillez entrer des valeurs valides pour tous les paramètres.")

# st.button("Start", key="verif", on_click=verification(model, train, n, iteration_count, alpha), type="secondary")

# Affichage des logs en temps réel
st_autorefresh(interval=2000, limit=None)
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'r') as log_file:
        log_content = log_file.read()
        st.text(log_content)

# Bouton pour nettoyer les logs
if st.button("Clean Logs"):
    clean_logs()
    st.experimental_rerun()
