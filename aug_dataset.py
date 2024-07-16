import os
import numpy as np
import albumentations as A
from PIL import Image
import cv2


def supprimer_images_augmentees(dossier):
    # Vérifier si le dossier existe
    if not os.path.exists(dossier):
        print(f"Le dossier {dossier} n'existe pas.")
        return

    # Compteur pour les fichiers supprimés
    compteur = 0

    # Parcourir tous les fichiers du dossier
    for nom_fichier in os.listdir(dossier):
        # Vérifier si le nom du fichier contient "_aug"
        if "_aug" in nom_fichier:
            chemin_fichier = os.path.join(dossier, nom_fichier)
            try:
                # Supprimer le fichier
                os.remove(chemin_fichier)
                compteur += 1
                print(f"Supprimé : {nom_fichier}")
            except Exception as e:
                print(f"Erreur lors de la suppression de {nom_fichier}: {e}")

    print(f"\nOpération terminée. {compteur} fichier(s) supprimé(s).")


def augment_images(input_folder, target_total=8000):
    # Définir les transformations d'augmentation
    transform = A.Compose([
        # Rotation aléatoire de l'image de 0, 90, 180 ou 270 degrés
        A.RandomRotate90(),

        # Retournement aléatoire de l'image (horizontal ou vertical)
        A.Flip(),

        # Transpose l'image (échange les axes)
        A.Transpose(),

        # Applique un type de bruit aléatoire avec une probabilité de 20%
        A.OneOf([
            A.GaussNoise(),  # Ajoute un bruit gaussien
            A.MultiplicativeNoise(),  # Ajoute un bruit multiplicatif
        ], p=0.2),

        # Applique un type de flou aléatoire avec une probabilité de 20%
        A.OneOf([
            A.MotionBlur(p=0.2),  # Flou de mouvement
            A.MedianBlur(blur_limit=3, p=0.1),  # Flou médian
            A.Blur(blur_limit=3, p=0.1),  # Flou gaussien
        ], p=0.2),

        # Combine le décalage, le changement d'échelle et la rotation
        # Limite de décalage : 6.25%, d'échelle : 20%, de rotation : 45°
        # Probabilité d'application : 20%
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),

        # Applique une distorsion aléatoire avec une probabilité de 20%
        A.OneOf([
            A.OpticalDistortion(p=0.3),  # Simule une distorsion optique
            A.GridDistortion(p=0.1),  # Applique une distorsion en grille
            A.ElasticTransform(p=0.3),  # Applique une transformation élastique
        ], p=0.2),

        # Applique un ajustement de contraste/netteté aléatoire avec une probabilité de 30%
        A.OneOf([
            A.CLAHE(clip_limit=2),  # Égalisation adaptative d'histogramme
            A.Sharpen(),  # Augmente la netteté
            A.Emboss(),  # Applique un effet d'embossage
            A.RandomBrightnessContrast(),  # Ajuste aléatoirement la luminosité et le contraste
        ], p=0.3),

        # Ajuste aléatoirement la teinte, la saturation et la valeur de l'image
        # Probabilité d'application : 30%
        A.HueSaturationValue(p=0.3),
    ])

    # Compter les images originales
    original_images = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    num_original = len(original_images)

    if num_original >= target_total:
        print(
            f"Le dossier contient déjà {num_original} images, ce qui est supérieur ou égal à la cible de {target_total}.")
        return

    # Calculer combien d'images augmentées sont nécessaires par image originale
    num_augmented_per_image = (target_total - num_original) // num_original
    remaining = (target_total - num_original) % num_original

    print(f"Nombre d'images originales: {num_original}")
    print(f"Nombre d'images augmentées à créer par image: {num_augmented_per_image}")
    print(f"Images supplémentaires nécessaires: {remaining}")

    for i, filename in enumerate(original_images):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Impossible de lire l'image: {filename}")
            continue

        # Convertir en RGB si l'image est en BGR
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            print(f"Format d'image non supporté pour {filename}: {img.shape}")
            continue

        # Déterminer combien d'images augmentées créer pour cette image
        num_to_generate = num_augmented_per_image + (1 if i < remaining else 0)

        # Générer des images augmentées
        for j in range(num_to_generate):
            augmented = transform(image=img)['image']
            augmented_img = Image.fromarray(augmented)

            base_filename, ext = os.path.splitext(filename)
            new_filename = f"{base_filename}_aug_{j}{ext}"
            augmented_img.save(os.path.join(input_folder, new_filename))

    print(f"Augmentation terminée. Le dossier contient maintenant {target_total} images.")


# Utilisation de la fonction
vache_dir = "./DataSet/vache"
chevre_dir = "./DataSet/chevre"
mouton_dir = "./DataSet/mouton"
augment_images(chevre_dir)

# supprimer_images_augmentees(mouton_dir)
