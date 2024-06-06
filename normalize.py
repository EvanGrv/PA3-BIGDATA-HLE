import os
from PIL import Image
import numpy as np


def normalize_image(image):
    # Convertir l'image en tableau numpy
    img_array = np.asarray(image, dtype=np.float32)
    # Normaliser les valeurs de pixels entre 0 et 1
    img_array /= 255.0
    # Convertir de nouveau en image PIL
    normalized_image = Image.fromarray((img_array * 255).astype(np.uint8))
    return normalized_image


def normalize_images(input_folder, output_folder):
    # Vérifier si le dossier d'entrée existe
    if not os.path.exists(input_folder):
        print(f"Le dossier {input_folder} n'existe pas.")
        return

    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialiser un compteur pour numéroter les images
    image_counter = 1

    # Parcourir tous les fichiers dans le dossier d'entrée
    for filename in os.listdir(input_folder):
        # Créer le chemin complet du fichier
        file_path = os.path.join(input_folder, filename)

        # Vérifier si c'est un fichier et non un dossier
        if os.path.isfile(file_path):
            try:
                # Ouvrir l'image
                img = Image.open(file_path)
                # Normaliser l'image
                normalized_img = normalize_image(img)

                # Obtenir l'extension de l'image originale
                file_extension = os.path.splitext(filename)[1]

                # Créer le nouveau nom de fichier avec la même extension
                new_filename = f"goat-{image_counter}{file_extension}"
                new_file_path = os.path.join(output_folder, new_filename)

                # Sauvegarder l'image normalisée dans le dossier de sortie
                normalized_img.save(new_file_path)
                print(f"Image {filename} normalisée et enregistrée en tant que {new_filename}.")

                # Incrémenter le compteur d'images
                image_counter += 1
            except Exception as e:
                print(f"Erreur lors du traitement de l'image {filename}: {e}")


# Chemins vers les dossiers d'entrée et de sortie
dossier_entree = "./DataSet/chevre"
dossier_sortie = "./DataSet/chevre_normalize"

# Appeler la fonction pour normaliser et renommer les images
normalize_images(dossier_entree, dossier_sortie)
