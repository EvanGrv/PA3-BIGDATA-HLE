import numpy as np
import ctypes
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import sys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Définir la structure Point
class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


# Charger la bibliothèque partagée de la librairie Rust
rbf_network_lib = ctypes.CDLL("../rbf/target/release/rbf.dll")

# Définition des types d'arguments et de la valeur de retour pour les fonctions de la librairie Rust
rbf_network_lib.rbf_network_new.restype = ctypes.c_void_p
rbf_network_lib.rbf_network_new.argtypes = [ctypes.c_size_t, ctypes.c_double, ctypes.c_size_t]
rbf_network_lib.rbf_network_fit.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(Point),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
rbf_network_lib.rbf_network_predict.restype = ctypes.POINTER(ctypes.c_double)
rbf_network_lib.rbf_network_predict.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(Point),
    ctypes.c_size_t,
]

# Définir le prototype de la fonction rbf_network_save
rbf_network_lib.rbf_network_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
rbf_network_lib.rbf_network_save.restype = None


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


def test_image(inputs, outputs, sigma, iteration_count):
    # Convertir les données en format compatible avec la librairie Rust
    points_array = (Point * len(inputs))()
    for i, (x, y) in enumerate(inputs):
        points_array[i].x = x
        points_array[i].y = y

    outputs_array = np.array(outputs, dtype=np.float64)
    outputs_c = outputs_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    num_units = len(inputs)

    # Créer une instance de RBFNetwork en utilisant la librairie Rust
    rbf_network = rbf_network_lib.rbf_network_new(num_units, sigma, len(outputs[0]))

    # Entraîner le réseau
    rbf_network_lib.rbf_network_fit(
        rbf_network,
        points_array,
        len(inputs),
        outputs_c,
        len(outputs),
        iteration_count,
        num_units,
        len(outputs[0])
    )

    model_name = f"./save_model/model_mlp_{sigma}_{iteration_count}.json"
    model_name_c = model_name.ctypes.data_as(ctypes.c_char_p)

    rbf_network_lib.rbf_network_save(rbf_network, model_name_c)

    # validate_model(mlp, k, inputs_test, npl, outputs_names, outputs_test)

    # save_path = f"./save_model/model_mlp_{layers}_{iteration_count}_{alpha}.json".encode('utf-8')
    # lib.save_mlp(mlp, save_path)
    #
    # lib.mlp_free(mlp)

    # # Fermeture de la barre de progression
    # tqdm_bar.close()



def test(X_train, Y_train, sigma, num_units, num_iterations):
    # Convertir les données en format compatible avec la librairie Rust
    points_array = (Point * len(X_train))()
    for i, (x, y) in enumerate(X_train):
        points_array[i].x = x
        points_array[i].y = y

    outputs_array = np.array(Y_train, dtype=np.float64)
    outputs_c = outputs_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Créer une instance de RBFNetwork en utilisant la librairie Rust

    rbf_network = rbf_network_lib.rbf_network_new(num_units, sigma, len(Y_train[0]))

    # Entraîner le réseau
    rbf_network_lib.rbf_network_fit(
        rbf_network,
        points_array,
        len(X_train),
        outputs_c,
        len(Y_train),
        num_iterations,
        num_units,
        len(Y_train[0])
    )

    # model_name = "test_rbf"
    # model_name_c = model_name.ctypes.data_as(ctypes.c_char_p)

    # rbf_network_lib.rbf_network_save(rbf_network, model_name_c)

    # Prédire les sorties pour les points d'entraînement
    grid_predictions1 = np.zeros((len(X_train), len(Y_train[0])))
    i = 0
    for i, (x, y) in enumerate(X_train):
        data_point = Point(x, y)
        prediction = rbf_network_lib.rbf_network_predict(
            rbf_network, ctypes.byref(data_point), len(Y_train[0])
        )
        prediction_array = np.ctypeslib.as_array(prediction, shape=(len(Y_train[0]),)).astype(float)
        grid_predictions1[i] = prediction_array
        print(prediction_array, "target", Y_train[i])

    # Prédire les sorties pour une grille de points
    x_min, x_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
    y_min, y_max = X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1
    step = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_predictions = np.zeros((len(grid_points), len(Y_train[0])))

    for i, (x, y) in enumerate(grid_points):
        data_point = Point(x, y)
        prediction = rbf_network_lib.rbf_network_predict(
            rbf_network, ctypes.byref(data_point), len(Y_train[0])
        )
        prediction_array = np.ctypeslib.as_array(prediction, shape=(len(Y_train[0]),)).astype(float)
        grid_predictions[i] = prediction_array

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
        plt.contourf(xx, yy, contour, levels=[-np.inf, 0.5, 1.5, np.inf], colors=['blue', 'red', 'green'], alpha=0.4)

    # Afficher le graphique
    plt.show()

    return xx, yy, grid_predictions, X_train, Y_train

    # Libérer la mémoire
    # rbf_network_lib.rbf_network_free(rbf_network)


def linear_simple():
    # test1 Linear Simple

    X = np.array([[1, 1], [2, 3], [3, 3]], dtype=np.float64)
    Y = np.array([[1], [-1], [-1]], dtype=np.float64)

    X_train = np.array(X, dtype=np.float64)
    Y_train = np.array(Y, dtype=np.float64)

    # Définir les paramètres du réseau RBF
    num_units = len(X_train)
    sigma = 0.1
    num_iterations = 10

    return test(X_train, Y_train, sigma, num_units, num_iterations)


def linear_multiple():
    # test2 Linear Multiple

    X = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    X_train = np.array(X, dtype=np.float64)
    Y_train = np.array(Y, dtype=np.float64)

    # Définir les paramètres du réseau RBF
    num_units = int(len(X_train) / 10)
    sigma = 0.1
    num_iterations = 10

    return test(X_train, Y_train, sigma, num_units, num_iterations)


def xor():
    # test 3 XOR

    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=np.float64)
    Y = np.array([[1], [1], [-1], [-1]], dtype=np.float64)

    X_train = np.array(X, dtype=np.float64)
    Y_train = np.array(Y, dtype=np.float64)

    # Définir les paramètres du réseau RBF
    num_units = len(X_train)
    sigma = 0.1
    num_iterations = 10

    return test(X_train, Y_train, sigma, num_units, num_iterations)


def cross():
    # test4 Cross

    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1] if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else [-1] for p in X], dtype=np.float64)

    X_train = np.array(X, dtype=np.float64)
    Y_train = np.array(Y, dtype=np.float64)

    # Définir les paramètres du réseau RBF
    num_units = int(len(X_train) / 10)
    sigma = 0.1
    num_iterations = 10

    return test(X_train, Y_train, sigma, num_units, num_iterations)


def multi_linear_classes():
    # test5 Multi Linear 3 classes

    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1, -1, -1] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                  [-1, 1, -1] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
                  [-1, -1, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
                  [0, 0, 0] for p in X], dtype=np.float64)

    X = X[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    Y = Y[[not np.all(arr == [0, 0, 0]) for arr in Y]]

    X_train = np.array(X, dtype=np.float64)
    Y_train = np.array(Y, dtype=np.float64)

    # Définir les paramètres du réseau RBF
    num_units = int(len(X_train) / 10)
    sigma = 0.1
    num_iterations = 10

    return test(X_train, Y_train, sigma, num_units, num_iterations)


def multi_cross():
    # test6 Multi Cross

    X = np.random.random((1000, 2)) * 2.0 - 1.0
    Y = np.array([[1, -1, -1] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [-1, 1, -1] if abs(
        p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [-1, -1, 1] for p in X])
    Y = np.array(Y, dtype=np.float64)

    X_train = np.array(X, dtype=np.float64)
    Y_train = np.array(Y, dtype=np.float64)

    # Définir les paramètres du réseau RBF
    num_units = int(len(X_train) / 10)
    sigma = 0.1
    num_iterations = 10

    return test(X_train, Y_train, sigma, num_units, num_iterations)


def cas_de_test():
    linear_simple()
    linear_multiple()
    xor()
    cross()
    multi_linear_classes()
    multi_cross()


# cas_de_test()
'''
#la regression :


inputs = np.array([
      [1.0],
      [2.0]
])
outputs = np.array([
      2.0,
      3.0
])


inputs = np.array([
      [1, 1],
      [2, 2],
      [3, 1]
],dtype=np.float64)

outputs = np.array([
      [2.0],
      [3.0],
      [2.5]
])



inputs= np.array([
      [1, 0],
      [0, 1],
      [1, 1],
      [0, 0],
],dtype=np.float64)
outputs = np.array([
      [2.0],
      [1.0],
      [-2.0],
      [-1.0]
],dtype=np.float64)

'''

def collecter_images(dossier, prefixe, liste_images):
    for fichier in os.listdir(dossier):
        if prefixe in fichier.lower():
            chemin_complet = os.path.join(dossier, fichier)
            if os.path.isfile(chemin_complet):
                liste_images.append(chemin_complet)


def test_train_image(iteration_count, sigma, class1, class2, class3):
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

    test_image(inputs_train, train_outputs, sigma, iteration_count)


iteration_count = 10000
sigma = 0.1


class1 = "../DataSet/vache"
class2 = "../DataSet/chevre"
class3 = "../DataSet/mouton"

# test_train_image(iteration_count, sigma, class1, class2, class3)