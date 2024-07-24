import ctypes
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from PIL import Image
import random
from tqdm import tqdm

# Chargement de la bibliothèque C
lib = ctypes.CDLL("..\\linear_model\\target\\release\\linear_model.dll")


# Définition de la structure LinearModel en Python
class LinearModel(ctypes.Structure):
    _fields_ = [
        ("weights", ctypes.POINTER(ctypes.c_double)),
        # ajuste la position verticale de la ligne de régression. Evite que la droite ne passe pas uniquement par l'origine
        ("bias", ctypes.POINTER(ctypes.c_double)),
        # mesure la différence entre les prédictions du modéle et les valeurs réelles
        ("loss", ctypes.POINTER(ctypes.c_double)),
        # nombres d'échantillons sur lesquels la perte a été calculée
        ("loss_size", ctypes.c_ulong),
    ]


# Définition de la signature de la fonction train_linear_model
lib.train_linear_model.restype = LinearModel  # type des valeurs de retour
lib.train_linear_model.argtypes = [  # type des valeurs en entrées
    ctypes.POINTER(ctypes.c_double),  # features
    ctypes.POINTER(ctypes.c_double),  # outputs
    ctypes.c_size_t,  # num_samples
    ctypes.c_size_t,  # num_features
    ctypes.c_double,  # learning_rate
    ctypes.c_size_t,  # num_iterations
    ctypes.c_size_t,  # k
    ctypes.c_bool,  # classification
]

# Définition de la signature de la fonction predict_linear_model
lib.predict_linear_model.restype = ctypes.POINTER(ctypes.c_double)
lib.predict_linear_model.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # features
    ctypes.POINTER(ctypes.c_double),  # weights
    ctypes.POINTER(ctypes.c_double),  # bias
    ctypes.c_size_t,  # num_samples
    ctypes.c_size_t,  # num_features
    ctypes.c_size_t,  # k
    ctypes.c_bool,  # classification
]


# Définir les types de retour et les arguments des fonctions Rust
lib.save_model_ml.argtypes = [ctypes.POINTER(LinearModel), ctypes.c_size_t, ctypes.c_char_p]
lib.save_model_ml.restype = ctypes.c_bool

lib.load_model_ml.argtypes = [ctypes.c_char_p]
lib.load_model_ml.restype = ctypes.POINTER(LinearModel)


def standardize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val == 0:
        # Si toutes les valeurs sont les mêmes, retourne une image de zéros
        n_image = np.zeros_like(image, dtype=np.float64)
    else:
        n_image = (image - min_val) / (max_val - min_val)
    return n_image


def read_image_as_1D(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_as_array = standardize_image(image)
    image_1D = image_as_array.flatten()
    return image_1D


def validate_linear_model(linear_model, inputs_test, outputs_test, outputs_names):
    predicted_outputs = []
    true_outputs = []

    weights = linear_model.weights
    bias = linear_model.bias
    num_samples = len(inputs_test)
    num_features = len(inputs_test[0])

    for i in range(num_samples):
        input_ptr = inputs_test[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        output_ptr = lib.predict_linear_model(input_ptr, weights, bias, 1, num_features, 3, ctypes.c_bool(True))

        predicted_output = np.array([output_ptr[j] for j in range(3)])  # 3 pour le nombre de classes
        predicted_class = np.argmax(predicted_output)
        true_class = np.argmax(outputs_test[i][0])

        predicted_outputs.append(predicted_class)
        true_outputs.append(true_class)

        print("Image:", outputs_names[i], "Predicted output:", predicted_output, "Result:", outputs_test[i])

    accuracy = np.mean(np.array(predicted_outputs) == np.array(true_outputs))
    print("Accuracy: ", accuracy)

    # Calculer et afficher la matrice de confusion
    class_names = ['vache', 'chevre', 'mouton']  # Assurez-vous que cet ordre correspond à vos classes
    cm = confusion_matrix(true_outputs, predicted_outputs)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='Blues')
    plt.title(f"Matrice de confusion (Accuracy: {accuracy:.2f})")

    # Enregistrer la figure sous forme de fichier JPEG
    save_path = f'./confusion_matrix/confusion_linear_{accuracy}_{alpha}_{iteration_count}.jpeg'
    plt.savefig(save_path, format='jpeg')

    plt.show()

def test(features, outputs, num_samples, num_features, learning_rate, num_iterations, k, classification):
    # Conversion des tableaux NumPy en tableaux C-style
    features_c = features.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    outputs_c = outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Appel de la fonction Rust
    linear_model = lib.train_linear_model(
        features_c,
        outputs_c,
        num_samples,
        num_features,
        learning_rate,
        num_iterations,
        k,
        ctypes.c_bool(classification),
    )

    # Accès aux tableaux de poids et de biais du modèle linéaire
    weights = np.ctypeslib.as_array(linear_model.weights, shape=(num_features * k,))
    bias = np.ctypeslib.as_array(linear_model.bias, shape=(k,))
    # Affichage des résultats
    print("Weights:", weights)
    print("Bias:", bias)

    xx, yy, grid_predictions = 0, 0, 0

    # Tracé de la séparation des classes
    if classification:
        if num_features == 3:
            # Tracé de la séparation des classes pour données 3D
            x_min, x_max = features[:, 0].min() - 0.1, features[:, 0].max() + 0.1
            y_min, y_max = features[:, 1].min() - 0.1, features[:, 1].max() + 0.1
            z_min, z_max = features[:, 2].min() - 0.1, features[:, 2].max() + 0.1
            step = 0.1

            xx, yy, zz = np.meshgrid(
                np.arange(x_min, x_max, step),
                np.arange(y_min, y_max, step),
                np.arange(z_min, z_max, step)
            )

            grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

            grid_predictions = np.zeros((len(grid_points), k))
            for i in range(len(grid_points)):
                grid_point_c = grid_points[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                grid_predictions_c = lib.predict_linear_model(
                    grid_point_c,
                    linear_model.weights,
                    linear_model.bias,
                    1,  # num_samples
                    num_features,
                    k,
                    ctypes.c_bool(classification),
                )
                grid_predictions[i] = np.ctypeslib.as_array(grid_predictions_c, shape=(k,))

            # Tracé des points d'entraînement avec des couleurs différentes pour chaque classe
            class_0 = features[outputs[:, 0] < 0]
            class_1 = features[outputs[:, 0] > 0]
            class_2 = features[np.argmax(outputs, axis=1) == 2]

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
                zz = -(weights[i * num_features] * xx + weights[i * num_features + 1] * yy + bias[i]) / weights[
                    i * num_features + 2]
                ax.plot_surface(xx, yy, zz, alpha=0.5)

            plt.legend()
            plt.show()

        elif num_features == 2:
            # Tracé de la séparation des classes pour données 2D
            x_min, x_max = features[:, 0].min() - 0.1, features[:, 0].max() + 0.1
            y_min, y_max = features[:, 1].min() - 0.1, features[:, 1].max() + 0.1
            step = 0.01

            xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            grid_predictions = np.zeros((len(grid_points), k))

            for i in range(len(grid_points)):
                grid_point_c = grid_points[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                grid_predictions_c = lib.predict_linear_model(
                    grid_point_c,
                    linear_model.weights,
                    linear_model.bias,
                    1,  # num_samples
                    num_features,
                    k,
                    ctypes.c_bool(classification),
                )
                grid_predictions[i] = np.ctypeslib.as_array(grid_predictions_c, shape=(k,))

            if k <= 2:
                # Tracé des points d'entraînement avec des couleurs différentes pour chaque classe
                class_0 = features[outputs[:, 0] < 0]
                class_1 = features[outputs[:, 0] > 0]

                plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', edgecolor='k', label='Classe 0')
                plt.scatter(class_1[:, 0], class_1[:, 1], color='red', edgecolor='k', label='Classe 1')

                # Tracer la séparation des classes
                contour = grid_predictions[:, 0].reshape(xx.shape)
                plt.contourf(xx, yy, contour, levels=[-np.inf, 0, np.inf], colors=['blue', 'red'], alpha=0.5)

            else:
                # Tracé des points d'entraînement avec des couleurs différentes pour chaque classe
                class_0 = features[outputs[:, 0] < 0]
                class_1 = features[outputs[:, 0] > 0]
                class_2 = features[np.argmax(outputs, axis=1) == 2]

                plt.scatter(class_0[:, 0], class_0[:, 1], color='red', edgecolor='k', label='Classe 0')
                plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', edgecolor='k', label='Classe 1')
                plt.scatter(class_2[:, 0], class_2[:, 1], color='green', edgecolor='k', label='Classe 2')

                # Tracer la séparation des classes
                contour = np.argmax(grid_predictions, axis=1).reshape(xx.shape)
                plt.contourf(xx, yy, contour, levels=[-np.inf, 0.5, 1.5, np.inf], colors=['blue', 'red', 'green'],
                             alpha=0.4)

            plt.legend()
            plt.show()

    else:
        # Régression linéaire

        if num_features == 1:
            # Régression linéaire avec pseudo-inverse de Moore Penrose pour une caractéristique
            ones = np.ones((features.shape[0], 1))
            X = np.hstack([ones, features])  # Ajout du biais
            Y = outputs

            # Calcul de la pseudo-inverse et des weights
            X_prime = np.linalg.pinv(X)
            w = X_prime.dot(Y)

            # Traçons les résultats en 2D
            plt.scatter(features, outputs, color='blue', label='Data points')
            x_vals = np.linspace(features.min(), features.max(), 100)
            y_vals = w[0] + w[1] * x_vals

            plt.plot(x_vals, y_vals, color='red', label='Regression line')
            plt.xlabel('Feature')
            plt.ylabel('Output')
            plt.legend()
            plt.show()


        elif num_features == 2:
            # Régression linéaire avec pseudo-inverse de Moore Penrose pour deux caractéristiques
            ones = np.ones((features.shape[0], 1))
            X = np.hstack([ones, features])
            Y = outputs

            # Calcul de la pseudo-inverse et des poids
            X_prime = np.linalg.pinv(X)
            w = X_prime.dot(Y)

            # Tracé des résultats en 3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(features[:, 0], features[:, 1], outputs, color='blue', label='Data points')

            x_surf, y_surf = np.meshgrid(np.linspace(features[:, 0].min(), features[:, 0].max(), 100),
                                         np.linspace(features[:, 1].min(), features[:, 1].max(), 100))
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

    return xx, yy, grid_predictions, features, outputs


def test_image(inputs, outputs, k, iteration_count, alpha, inputs_test, test_outputs, test_images):
    inputs_c = inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    outputs_c = outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    num_samples, num_inputs = inputs.shape

    # Appel de la fonction Rust
    linear_model = lib.train_linear_model(
        inputs_c,
        outputs_c,
        num_samples,
        num_inputs,
        alpha,
        iteration_count,
        k,
        ctypes.c_bool(True),
    )

    validate_linear_model(linear_model, inputs_test, test_outputs, test_images)

    save_path = f"./save_model/model_ml_{iteration_count}_{alpha}_{k}.json".encode('utf-8')
    lib.save_model_ml(linear_model, num_inputs, save_path)

    # Fermeture de la barre de progression
    # tqdm_bar.close()


# Exemple de Classification

'''
num_samples, num_features = inputs.shape
learning_rate = 0.001
num_iterations = 100000
k = 1

test(inputs, outputs, num_samples, num_features, learning_rate, num_iterations, k, True)
'''


def linear_simple():
    # test 1: Linear Model
    inputs = np.array([[1, 1], [2, 3], [3, 3]], dtype=np.float64)
    outputs = np.array([[1], [-1], [-1]], dtype=np.float64)

    num_samples, num_features = inputs.shape
    learning_rate = 0.001
    num_iterations = 100000
    k = 1

    return test(inputs, outputs, num_samples, num_features, learning_rate, num_iterations, k, True)


def linear_multiple():
    # test 2: Linear Multiple
    inputs = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    outputs = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    num_samples, num_features = inputs.shape
    learning_rate = 0.001
    num_iterations = 100000
    k = 1

    return test(inputs, outputs, num_samples, num_features, learning_rate, num_iterations, k, True)


def xor():
    # test 3: XOR
    # Ne fonctionne pas pour le modèle linéaire
    inputs = np.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=np.float64)
    outputs = np.array([1, 1, -1, -1], dtype=np.float64)

    num_samples, num_features = inputs.shape
    learning_rate = 0.001
    num_iterations = 100000
    k = 2

    return test(inputs, outputs, num_samples, num_features, learning_rate, num_iterations, k, True)


def cross():
    # test 4: CROSS
    inputs = np.random.random((500, 2)) * 2.0 - 1.0
    outputs = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in inputs])
    num_samples, num_features = inputs.shape
    learning_rate = 0.001
    num_iterations = 100000
    k = 1

    return test(inputs, outputs, num_samples, num_features, learning_rate, num_iterations, k, True)


def multi_linear_classes():
    # test 5: Multi Linear 3 class
    inputs = np.random.random((500, 2)) * 2.0 - 1.0
    outputs = np.array([[1, -1, -1] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                        [-1, 1, -1] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else [-1, -1,
                                                                                                           1] if -p[0] -
                                                                                                                 p[
                                                                                                                     1] - 0.5 < 0 and
                                                                                                                 p[
                                                                                                                     1] < 0 and
                                                                                                                 p[0] -
                                                                                                                 p[
                                                                                                                     1] - 0.5 > 0 else
                        [0, 0, 0] for p in inputs], dtype=np.float64)

    inputs = inputs[[not np.all(arr == [0, 0, 0]) for arr in outputs]]
    outputs = outputs[[not np.all(arr == [0, 0, 0]) for arr in outputs]]

    num_samples, num_features = inputs.shape
    learning_rate = 0.005
    num_iterations = 100000
    k = 3

    return test(inputs, outputs, num_samples, num_features, learning_rate, num_iterations, k, True)


def multi_cross():
    # test 7: Multi Cross
    inputs = np.random.random((1000, 2)) * 2.0 - 1.0
    outputs = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [0, 1, 0] if abs(
        p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [0, 0, 1] for p in inputs])

    num_samples, num_features = inputs.shape
    learning_rate = 0.005
    num_iterations = 100000
    k = 3

    return test(inputs, outputs, num_samples, num_features, learning_rate, num_iterations, k, True)


def cas_de_test():
    linear_simple()
    linear_multiple()
    #xor()
    #cross()
    multi_linear_classes()
    multi_cross()


# Exemple de Régression
# Paramètres à utiliser:

# Test 1
def linear_simple_2D():
    inputs = np.array([
        [5.0],
        [2.0]
    ])
    outputs = np.array([
        4.0,
        6.0
    ])

    num_samples, num_features = inputs.shape
    learning_rate = 0.001
    num_iterations = 100000
    k = 1

    return test(inputs, outputs, num_samples, num_features, learning_rate, num_iterations, k, False)


# Test 2
def non_linear_simple_2D():
    inputs = np.array([
        [1],
        [2],
        [3]
    ])
    outputs = np.array([
        2,
        3,
        2.5
    ])

    num_samples, num_features = inputs.shape
    learning_rate = 0.001
    num_iterations = 100000
    k = 1

    return test(inputs, outputs, num_samples, num_features, learning_rate, num_iterations, k, False)


#Test 3
def linear_simple_3D():
    inputs = np.array([
        [1, 1],
        [2, 2],
        [3, 1]
    ])
    outputs = np.array([
        2,
        3,
        2.5
    ])

    num_samples, num_features = inputs.shape
    learning_rate = 0.001
    num_iterations = 10000000
    k = 1

    return test(inputs, outputs, num_samples, num_features, learning_rate, num_iterations, k, False)


#Test 4
def linear_tricky_3D():
    inputs = np.array([
        [1, 1],
        [2, 2],
        [3, 3]
    ])
    outputs = np.array([
        1,
        2,
        3
    ])

    num_samples, num_features = inputs.shape
    learning_rate = 0.001
    num_iterations = 10000
    k = 1

    return test(inputs, outputs, num_samples, num_features, learning_rate, num_iterations, k, False)


#Test 5
def non_linear_tricky_3D():
    inputs = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ])
    outputs = np.array([
        2,
        1,
        -2,
        -1
    ])

    num_samples, num_features = inputs.shape
    learning_rate = 0.001
    num_iterations = 100000
    k = 1

    return test(inputs, outputs, num_samples, num_features, learning_rate, num_iterations, k, False)


def cas_de_test_régression():
    linear_simple_2D()
    non_linear_simple_2D()
    linear_simple_3D()
    linear_tricky_3D()
    non_linear_tricky_3D()


# cas_de_test()
# cas_de_test_régression()


def collecter_images(dossier, prefixe, liste_images):
    for fichier in os.listdir(dossier):
        if prefixe in fichier.lower():
            chemin_complet = os.path.join(dossier, fichier)
            if os.path.isfile(chemin_complet):
                liste_images.append(chemin_complet)


def test_train_image(iteration_count, alpha, class1, class2, class3):
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

    test_image(inputs_train, train_outputs, 3, iteration_count, alpha, inputs_test, test_outputs, test_images)


iteration_count = 1000000
alpha = 0.001

class1 = "../DataSet/vache"
class2 = "../DataSet/chevre"
class3 = "../DataSet/mouton"

test_train_image(iteration_count, alpha, class1, class2, class3)
