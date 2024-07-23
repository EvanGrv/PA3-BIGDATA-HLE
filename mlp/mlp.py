import ctypes
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json

from PIL import Image
import random
from tqdm import tqdm

from mpl_toolkits.mplot3d import Axes3D

lib = ctypes.CDLL("../mlp/target/release/mlp.dll")

# Définir le type de fonction callback
ProgressCallbackType = ctypes.CFUNCTYPE(None, ctypes.c_double, ctypes.c_void_p)

# Initialisation de tqdm pour la barre de progression
tqdm_bar = tqdm(total=100, desc="Training Progress", unit="%", position=0, mininterval=0.1)


# Fonction callback
def progress_callback(progress, user_data):
    tqdm_bar.update(progress - tqdm_bar.n)  # Mise à jour de l'avancement
    tqdm_bar.set_description(f"Training Progress: {progress:.2f}%")  # Mise à jour du libellé
    sys.stdout.flush()


# Convertir la fonction callback en un type compatible C
progress_callback_c = ProgressCallbackType(progress_callback)

# Définir la structure MLP en Python
lib.mlp_predict.restype = ctypes.POINTER(ctypes.c_double)
lib.mlp_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_uint, ctypes.c_bool]
lib.mlp_free.argtypes = [ctypes.c_void_p]
lib.train_mlp.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_uint,
    ctypes.c_bool,
    ctypes.c_size_t,
    ctypes.c_double,
    ProgressCallbackType,
    ctypes.c_void_p
]

lib.load_mlp.restype = ctypes.c_void_p
lib.load_mlp.argtypes = [ctypes.c_char_p]

lib.save_mlp.restype = None
lib.save_mlp.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

# Définition de la signature de la fonction create_mlp
lib.create_mlp.argtypes = (ctypes.POINTER(ctypes.c_uint), ctypes.c_size_t)
lib.create_mlp.restype = ctypes.c_void_p


def lire_fichier_json(nom_fichier):
    with open(nom_fichier, 'r') as fichier:
        contenu = json.load(fichier)
    return contenu


def load_images_from_directory(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img = Image.open(os.path.join(directory, filename)).convert('L')  # Convertir en niveaux de gris
            img = np.array(img).flatten()  # Aplatir l'image
            images.append(img)
            # Extraire le label du nom du fichier (par exemple, "goat-1.png" où "goat" est le label)
            label = filename.split('-')[0]
            labels.append(label)
    return np.array(images), np.array(labels)


def standardize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val == 0:
        # Si toutes les valeurs sont les mêmes, retourne une image de zéros
        n_image = np.zeros_like(image, dtype=np.float64)
    else:
        n_image = (image - min_val) / (max_val - min_val)
    return n_image


def get_flattened_image_size(image_path):
    # Charger l'image
    image = Image.open(image_path)

    # Convertir en RGB si nécessaire (pour s'assurer qu'il y a 3 canaux)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Obtenir les dimensions de l'image
    width, height = image.size
    num_channels = len(image.getbands())  # Normalement 3 pour RGB

    # Calculer la taille de l'image aplatie
    flattened_size = width * height * num_channels

    return flattened_size


def read_image_as_1D(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_as_array = standardize_image(image)
    image_1D = image_as_array.flatten()
    return image_1D


def validate_model(mlp, k, inputs_test, npl, outputs_names, outputs_test):
    # Prédiction avec le MLP entraîné
    if (k <= 2):
        predicted_outputs = []
        for i in range(len(inputs_test)):
            input_ptr = inputs_test[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            output_ptr = lib.mlp_predict(mlp, input_ptr, len(inputs_test[i]), ctypes.c_bool(True))
            predicted_output = np.array([output_ptr[j] for j in range(npl[-1])])
            lib.mlp_free(output_ptr)
            predicted_outputs.append(-1 if predicted_output < 0 else 1)

            print("Image:", outputs_names[i], "Predicted output:", predicted_output, "resulat", outputs_test[i])

    else:
        predicted_outputs = []
        correct_outputs = []
        true_outputs = []

        print("len inputs_test:", len(inputs_test))

        for i in range(len(inputs_test)):
            print(i)
            input_ptr = inputs_test[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            output_ptr = lib.mlp_predict(mlp, input_ptr, len(inputs_test[i]), ctypes.c_bool(True))
            predicted_output = np.array([-1 if output_ptr[j] < 0 else 1 for j in range(npl[-1])])
            to_show = np.array([output_ptr[j] for j in range(npl[-1])])
            # comparison entre predicted_output et outputs_test[i]
            # si on ets bon correct_output.append(1) sinon on fait rien
            predicted_output = predicted_output.astype(np.float64)

            true_class = np.argmax(outputs_test[i][0])

            true_outputs.append(true_class)

            if np.array_equal(predicted_output, outputs_test[i][0]):
                correct_outputs.append(outputs_names[i])

            lib.mlp_free(output_ptr)
            predicted_outputs.append(np.argmax(predicted_output))
            print("Image:", outputs_names[i], "Predicted output:", to_show, "resulat", outputs_test[i])

        accuracy = len(correct_outputs) / len(outputs_test)
        print("Accuracy: ", accuracy)

        # Calculer et afficher la matrice de confusion
        class_names = ['vache', 'chevre', 'mouton']  # Assurez-vous que cet ordre correspond à vos classes
        cm = confusion_matrix(true_outputs, predicted_outputs)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, cmap='Blues')
        plt.title(f"Matrice de confusion (Accuracy: {accuracy:.2f})")

        # Enregistrer la figure sous forme de fichier JPEG
        save_path = f'./confusion_matrix/confusion_mlp_{accuracy}_{npl}.jpeg'
        plt.savefig(save_path, format='jpeg')

        plt.show()


def test(inputs, outputs, n, title, k, iteration_count, alpha, isclassification):
    # Définition des paramètres
    npl = np.array(n)

    # Conversion du tableau npl en tableau de type ctypes.c_uint
    npl_array = (ctypes.c_uint * len(npl))(*npl)

    # Création du MLP
    mlp = lib.create_mlp(npl_array, len(npl))

    # Entraînement du MLP
    lib.train_mlp(mlp, inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(inputs),
                  outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(outputs),
                  ctypes.c_bool(isclassification), iteration_count, alpha, progress_callback_c,
                  None)

    predicted_outputs1 = []
    for i in range(len(inputs)):
        input_ptr = inputs[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        output_ptr = lib.mlp_predict(mlp, input_ptr, len(inputs[i]), ctypes.c_bool(isclassification))
        predicted_output = np.array([output_ptr[j] for j in range(npl[-1])])
        lib.mlp_free(output_ptr)
        predicted_outputs1.append(predicted_output)

        print("Input:", inputs[i], "Predicted output:", predicted_output, "resulat", outputs[i])

    # Prédiction avec le MLP entraîné
    print(title)

    predicted_outputs = []
    num_feature = inputs.shape[1]

    x_min, x_max, y_min, y_max = 0, 0, 0, 0
    xx, yy = 0, 0
    step = 0.01

    grid_points = 0
    grid_predictions = 0

    if isclassification:
        x_min, x_max = inputs[:, 0].min() - 0.1, inputs[:, 0].max() + 0.1
        y_min, y_max = inputs[:, 1].min() - 0.1, inputs[:, 1].max() + 0.1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_predictions = np.zeros((len(grid_points), k))  # Update this line

        if num_feature == 3:
            # Tracé de la séparation des classes pour données 3D
            z_min, z_max = inputs[:, 2].min() - 0.1, inputs[:, 2].max() + 0.1

            zz = np.meshgrid(
                np.arange(z_min, z_max, step)
            )

            grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
            grid_predictions = np.zeros((len(grid_points), k))

            for i in range(len(grid_points)):
                input_ptr = grid_points[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                output_ptr = lib.mlp_predict(mlp, input_ptr, len(grid_points[i]),
                                             ctypes.c_bool(isclassification))
                predicted_output = np.array([output_ptr[j] for j in range(npl[-1])])
                lib.mlp_free(output_ptr)
                grid_predictions[i] = np.ctypeslib.as_array(predicted_output, shape=(k,))

            # Tracé des points d'entraînement avec des couleurs différentes pour chaque classe
            class_0 = inputs[outputs[:, 0] < 0]
            class_1 = inputs[outputs[:, 0] > 0]
            class_2 = inputs[np.argmax(outputs, axis=1) == 2]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(class_0[:, 0], class_0[:, 1], class_0[:, 2], color='blue', edgecolor='k', label='Classe 0')
            ax.scatter(class_1[:, 0], class_1[:, 1], class_1[:, 2], color='red', edgecolor='k', label='Classe 1')
            ax.scatter(class_2[:, 0], class_2[:, 1], class_2[:, 2], color='green', edgecolor='k', label='Classe 2')

            # Calcul de la ligne de séparation pour un modèle 3D
            x_vals = np.array([x_min, x_max])
            y_vals = np.array([y_min, y_max])

            for i in range(k):
                xx, yy = np.meshgrid(x_vals, y_vals)
                ax.plot_surface(xx, yy, zz, alpha=0.5)

            plt.legend()
            plt.show()

        elif num_feature == 2:
            for i in range(len(grid_points)):
                input_ptr = grid_points[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                output_ptr = lib.mlp_predict(mlp, input_ptr, len(grid_points[i]), ctypes.c_bool(isclassification))
                predicted_output = np.array([output_ptr[j] for j in range(npl[-1])])
                lib.mlp_free(output_ptr)
                grid_predictions[i] = np.ctypeslib.as_array(predicted_output, shape=(k,))

            if k <= 2:
                # Conversion des prédictions en couleurs pour le tracé du graphe
                class_0 = inputs[outputs[:, 0] < 0]
                class_1 = inputs[outputs[:, 0] > 0]

                plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', edgecolor='k', label='Classe 0')
                plt.scatter(class_1[:, 0], class_1[:, 1], color='red', edgecolor='k', label='Classe 1')

                # Tracer la séparation des classes
                contour = grid_predictions[:, 0].reshape(xx.shape)
                plt.contourf(xx, yy, contour, levels=[-np.inf, 0, np.inf], colors=['blue', 'red'], alpha=0.5)
                plt.title(title)
                # Affichage du graphe
                plt.show()
            else:
                class_0 = inputs[outputs[:, 0] < 0]
                class_1 = inputs[outputs[:, 0] > 0]
                class_2 = inputs[np.argmax(outputs, axis=1) == 2]

                # Tracé des points d'entraînement
                plt.scatter(class_0[:, 0], class_0[:, 1], color='red', edgecolor='k', label='Classe 0')
                plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', edgecolor='k', label='Classe 1')
                plt.scatter(class_2[:, 0], class_2[:, 1], color='green', edgecolor='k', label='Classe 2')

                print(grid_predictions)
                contour = np.argmax(grid_predictions, axis=1).reshape(xx.shape)
                plt.contourf(xx, yy, contour, levels=[-np.inf, 0.5, 1.5, np.inf], colors=['blue', 'red', 'green'],
                             alpha=0.4)

                plt.title(title)
                # Affichage du graphe
                plt.show()

    else:

        '''predicted_outputs = []

        fichier_json = "./mlp_model.json"

        # Convertir les listes en tableaux numpy avec dtype=object

        mlp_from_file = lib.load_mlp(fichier_json.encode('utf-8'))'''

        ''' for i in range(len(grid_points)):
            input_ptr = grid_points[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            output_ptr = lib.mlp_predict(mlp, input_ptr, len(grid_points[i]),
                                         ctypes.c_bool(isclassification))
            predicted_output = np.array([output_ptr[j] for j in range(npl[-1])])
            lib.mlp_free(output_ptr)
            grid_predictions[i] = np.ctypeslib.as_array(predicted_output, shape=(k,))

            print("Input:", inputs[i], "Predicted output:", predicted_output, "resulat", outputs[i])

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(inputs[:, 0], inputs[:, 1], predicted_output)
        plt.show()
        plt.clf()'''

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
            plt.show()


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
            plt.show()

        else:
            raise ValueError(
                "Ce code ne gère actuellement que la régression linéaire pour jusqu'à deux caractéristiques")

    lib.mlp_free(mlp)
    return xx, yy, grid_predictions, inputs, outputs


def test_image(inputs, outputs, layers, title, k, iteration_count, alpha, images_name, inputs_test, outputs_test,
               outputs_names):
    # Définition des paramètres
    npl = np.array(layers)

    # Conversion du tableau npl en tableau de type ctypes.c_uint
    npl_array = (ctypes.c_uint * len(npl))(*npl)

    # Création du MLP
    mlp = lib.create_mlp(npl_array, len(npl))
    # mlp = lib.load_mlp("./save_model/model_mlp_[4800, 256, 128, 3]_1000_0.001.json".encode('utf-8'))

    #Entraînement du MLP
    lib.train_mlp(mlp, inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(inputs),
                  outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(outputs),
                  ctypes.c_bool(True), iteration_count, alpha, progress_callback_c,
                  None)

    validate_model(mlp, k, inputs_test, npl, outputs_names, outputs_test)

    print("save")
    save_path = f"./save_model/model_mlp_{layers}_{iteration_count}_{alpha}.json".encode('utf-8')
    lib.save_mlp(mlp, save_path)

    lib.mlp_free(mlp)

    # Fermeture de la barre de progression
    tqdm_bar.close()


def linear_simple():
    # test1 Linear Simple

    X = np.array([[1, 1], [2, 3], [3, 3]], dtype=np.float64)
    Y = np.array([[1], [-1], [-1]], dtype=np.float64)

    arr = [2, 1]

    alpha = 0.01
    nb_iter = 100000
    is_classification = True

    return test(X, Y, arr, "test1", 2, nb_iter, alpha, is_classification)


def linear_multiple():
    # test2 Linear Multiple

    X = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    arr = [2, 1]

    alpha = 0.01
    nb_iter = 100000
    is_classification = True

    return test(X, Y, arr, "test2", 2, nb_iter, alpha, is_classification)


def xor():
    # test 3 XOR

    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=np.float64)
    Y = np.array([[1], [1], [-1], [-1]], dtype=np.float64)

    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)

    arr = [2, 2, 1]

    alpha = 0.001
    nb_iter = 1000000
    is_classification = True

    return test(X, Y, arr, "test3", 2, nb_iter, alpha, is_classification)


def cross():
    # test4 Cross

    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1] if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else [-1] for p in X], dtype=np.float64)

    arr = [2, 4, 1]

    alpha = 0.01
    nb_iter = 1000000
    is_classification = True

    return test(X, Y, arr, "test4", 2, nb_iter, alpha, is_classification)


def multi_linear_classes():
    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1, -1, -1] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                  [-1, 1, -1] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
                  [-1, -1, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
                  [0, 0, 0] for p in X], dtype=np.float64)

    X = X[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    Y = Y[[not np.all(arr == [0, 0, 0]) for arr in Y]]

    arr = [2, 3]

    alpha = 0.1
    nb_iter = 100000
    is_classification = True

    return test(X, Y, arr, "test5", 3, nb_iter, alpha, is_classification)


def multi_cross():
    # test6 Multi Cross

    X = np.random.random((1000, 2)) * 2.0 - 1.0
    Y = np.array([[1, -1, -1] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [-1, 1, -1] if abs(
        p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [-1, -1, 1] for p in X])
    Y = np.array(Y, dtype=np.float64)

    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)

    arr = [2, 100, 100, 3]

    alpha = 0.001
    nb_iter = 1000000
    is_classification = True

    return test(X, Y, arr, "test6", 3, nb_iter, alpha, is_classification)


def cas_de_test():
    linear_simple()
    linear_multiple()
    xor()
    cross()
    multi_linear_classes()
    multi_cross()


# Exemple de Régression
# Paramètres à utiliser:

# Test 1
def linear_simple_2D():
    X = np.array([
        [5.0],
        [2.0]
    ])
    Y = np.array([
        4.0,
        6.0
    ])

    arr = [1, 1]

    alpha = 0.001
    nb_iter = 100000
    is_classification = False
    k = 1

    return test(X, Y, arr, "test1", k, nb_iter, alpha, is_classification)


# Test 2
def non_linear_simple_2D():
    X = np.array([
        [1],
        [2],
        [3]
    ])
    Y = np.array([
        2,
        3,
        2.5
    ])

    arr = [1, 2, 1]

    alpha = 0.001
    nb_iter = 100000
    is_classification = False
    k = 1

    return test(X, Y, arr, "test2", k, nb_iter, alpha, is_classification)


# Test 3
def linear_simple_3D():
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 1]
    ])
    Y = np.array([
        2,
        3,
        2.5
    ])

    arr = [2, 1]

    alpha = 0.001
    nb_iter = 100000
    is_classification = False
    k = 1

    return test(X, Y, arr, "test3", k, nb_iter, alpha, is_classification)


# Test 4
def linear_tricky_3D():
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 3]
    ])
    Y = np.array([
        1,
        2,
        3
    ])

    arr = [2, 1]

    alpha = 0.001
    nb_iter = 100000
    is_classification = False
    k = 1

    return test(X, Y, arr, "test4", k, nb_iter, alpha, is_classification)


# Test 5
def non_linear_tricky_3D():
    X = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ])
    Y = np.array([
        2,
        1,
        -2,
        -1
    ])

    arr = [2, 2, 1]

    alpha = 0.001
    nb_iter = 100000
    is_classification = False
    k = 1

    return test(X, Y, arr, "test5", k, nb_iter, alpha, is_classification)


def cas_de_test_régression():
    linear_simple_2D()
    non_linear_simple_2D()
    linear_simple_3D()
    linear_tricky_3D()
    non_linear_tricky_3D()


#cas_de_test()
#cas_de_test_régression()

def predict_image_class(mlp, image_path, npl, class_names):
    # Charger et standardiser l'image
    image_1D = read_image_as_1D(image_path)

    # Convertir l'image en pointeur C
    input_ptr = image_1D.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Appeler la fonction de prédiction du modèle MLP
    output_ptr = lib.mlp_predict(mlp, input_ptr, len(image_1D), ctypes.c_bool(True))

    # Convertir la sortie en tableau NumPy
    predicted_output = np.array([output_ptr[j] for j in range(npl[-1])])
    predicted_output = predicted_output.astype(np.float64)

    # Libérer la mémoire allouée pour la sortie du modèle
    lib.mlp_free(output_ptr)

    # Trouver la classe prédite
    predicted_class_index = np.argmax(predicted_output)
    predicted_class = class_names[predicted_class_index]

    return predicted_class, predicted_output


def collecter_images(dossier, prefixe, liste_images):
    for fichier in os.listdir(dossier):
        if prefixe in fichier.lower():
            chemin_complet = os.path.join(dossier, fichier)
            if os.path.isfile(chemin_complet):
                liste_images.append(chemin_complet)


def test_train_image(n, iteration_count, alpha, class1, class2, class3):
    vache_dir = os.path.normpath(class1)
    chevre_dir = os.path.normpath(class2)
    mouton_dir = os.path.normpath(class3)

    vache_images = []
    chevre_images = []
    mouton_images = []

    # Collecte des images pour chaque catégorie
    collecter_images(vache_dir, "vache", vache_images)
    collecter_images(chevre_dir, "goat", chevre_images)
    collecter_images(mouton_dir, "mouton", mouton_images)

    # Mélanger les images
    random.shuffle(vache_images)
    random.shuffle(chevre_images)
    random.shuffle(mouton_images)

    print("image collected")

    # ratio d'images dans le dataset de train et de validation
    train_ratio = 0.8
    # test_ratio = 0.2

    vache_train_count = int(len(vache_images) * train_ratio)
    chevre_train_count = int(len(chevre_images) * train_ratio)
    mouton_train_count = int(len(mouton_images) * train_ratio)

    # vache_test_count = len(vache_images) - vache_train_count
    # chevre_test_count = len(chevre_images) - chevre_train_count
    # mouton_test_count = len(mouton_images) - mouton_train_count

    vache_train_images = vache_images[:vache_train_count]
    vache_test_images = vache_images[vache_train_count:]

    chevre_train_images = chevre_images[:chevre_train_count]
    chevre_test_images = chevre_images[chevre_train_count:]

    mouton_train_images = mouton_images[:mouton_train_count]
    mouton_test_images = mouton_images[mouton_train_count:]

    train_images = vache_train_images + chevre_train_images + mouton_train_images
    test_images = vache_test_images + chevre_test_images + mouton_test_images

    train_outputs = []
    test_outputs = []

    for image_path in train_images:
        # determiner la classe de l'image à partir du path de l'image
        if "vache" in image_path:
            label = [1, -1, -1]
        elif "chevre" in image_path:
            label = [-1, 1, -1]
        elif "mouton" in image_path:
            label = [-1, -1, 1]

        # creer le vecteur output
        output_vector = [label]

        # ajouter le vecteur output de l'image dans le vecteur output de dataset d'apprentissage
        train_outputs.append(output_vector)

    for image_path in test_images:
        if "vache" in image_path:
            label = [1, -1, -1]
        elif "chevre" in image_path:
            label = [-1, 1, -1]
        elif "mouton" in image_path:
            label = [-1, -1, 1]

        output_vector = [label]

        test_outputs.append(output_vector)

    inputs_train = [read_image_as_1D(img_path) for img_path in train_images]
    inputs_train = np.array(inputs_train)

    inputs_test = [read_image_as_1D(img_path) for img_path in test_images]
    inputs_test = np.array(inputs_test)

    train_images = np.array(train_images)

    train_outputs = np.array(train_outputs, dtype=np.float64)

    test_outputs = np.array(test_outputs, dtype=np.float64)

    test_image(inputs_train, train_outputs, n, "Training Results", 3, iteration_count, alpha,
               train_images, inputs_test, test_outputs, test_images)


# n = [4800, 512, 256, 3]
n = [4800, 102, 102, 3]
iteration_count = 20000
alpha = 0.0001

class1 = "../DataSet/vache"
class2 = "../DataSet/chevre"
class3 = "../DataSet/mouton"

test_train_image(n, iteration_count, alpha, class1, class2, class3)
