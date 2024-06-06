import os
from PIL import Image


def resize_and_rename_images_in_folder(folder_path, size=(40, 40)):
    # Vérifier si le dossier existe
    if not os.path.exists(folder_path):
        print(f"Le dossier {folder_path} n'existe pas.")
        return

    # Initialiser un compteur pour numéroter les images
    image_counter = 1

    # Parcourir tous les fichiers dans le dossier
    for filename in os.listdir(folder_path):
        # Créer le chemin complet du fichier
        file_path = os.path.join(folder_path, filename)

        # Vérifier si c'est un fichier et non un dossier
        if os.path.isfile(file_path):
            try:
                # Ouvrir l'image
                img = Image.open(file_path)
                # Redimensionner l'image
                img = img.resize(size, Image.ANTIALIAS)

                # Obtenir l'extension de l'image originale
                file_extension = os.path.splitext(filename)[1]

                # Créer le nouveau nom de fichier avec la même extension
                new_filename = f"goat-{image_counter}{file_extension}"
                new_file_path = os.path.join(folder_path, new_filename)

                # Sauvegarder l'image redimensionnée avec le nouveau nom
                img.save(new_file_path)
                print(f"Image {filename} redimensionnée et renommée en {new_filename}.")

                # Supprimer l'ancien fichier
                os.remove(file_path)

                # Incrémenter le compteur d'images
                image_counter += 1
            except Exception as e:
                print(f"Erreur lors du traitement de l'image {filename}: {e}")


# Chemin vers le dossier contenant les images
dossier_images = "./DataSet/chevre"

# Appeler la fonction pour redimensionner et renommer les images
resize_and_rename_images_in_folder(dossier_images)
