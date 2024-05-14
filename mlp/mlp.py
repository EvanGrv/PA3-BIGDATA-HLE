import numpy as np
import ctypes
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

lib = ctypes.CDLL("target/debug/mlp.dll")


# Définir la structure MLP en Python
class MLP(ctypes.Structure):
    _fields_ = [
        ("d", ctypes.POINTER(ctypes.c_size_t)),
        ("W", ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))),
        ("L", ctypes.c_size_t),
        ("X", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
        ("deltas", ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
    ]


# Définir les types de retour et les arguments de vos fonctions Rust
lib.create_mlp_model.restype = ctypes.POINTER(MLP)
lib.create_mlp_model.argtypes = [ctypes.POINTER(ctypes.c_longlong), ctypes.c_size_t]

# Fonction train_mlp_model
lib.train_mlp_model.argtypes = [ctypes.POINTER(MLP),
                                ctypes.POINTER(ctypes.c_float),
                                ctypes.c_int,
                                ctypes.c_int,
                                ctypes.POINTER(ctypes.c_float),
                                ctypes.c_int,
                                ctypes.c_float,
                                ctypes.c_int,
                                ctypes.c_bool]

# Fonction predict_mlp_model
lib.predict_mlp_model.restype = ctypes.POINTER(ctypes.c_float)
lib.predict_mlp_model.argtypes = [ctypes.POINTER(MLP),
                                  ctypes.POINTER(ctypes.c_float),
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_bool]

# Fonction delete_mlp_model
lib.delete_mlp_model.argtypes = [ctypes.POINTER(MLP)]


def plot_classification(X, predictions, colors):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    # Générer une grille de points couvrant tout l'espace 2D
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    # Utiliser les prédictions pour définir les régions de décision
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], predictions[:2]) + predictions[2]
    Z = Z.reshape(xx.shape)
    Z_class = Z > 0  # Déterminer les classes sur la grille

    # Tracer les régions de décision en arrière-plan
    cmap_background = ListedColormap(colors)
    plt.contourf(xx, yy, Z_class, alpha=0.8, cmap=cmap_background)

    # Tracer les points de données
    plt.scatter(X[:, 0], X[:, 1],
                c=['blue' if pred > 0 else 'red' for pred in np.dot(X, predictions[:2]) + predictions[2]],
                edgecolor='k')

    # Afficher le graphique
    plt.show()


def test_train():
    X = np.array([[
        [1, 1],
        [2, 3],
        [3, 3]
    ]], dtype=np.float64)

    Y = np.array([[
        [1],
        [-1],
        [-1]
    ]], dtype=np.float64)

    arr = [2, 1]

    arr_ptr = (ctypes.c_longlong * len(arr))(*arr)
    model_ptr = lib.create_mlp_model(arr_ptr, len(arr))

    # Appeler la fonction train_mlp_model pour entraîner le modèle
    lib.train_mlp_model(model_ptr,
                        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  # données d'entrée
                        ctypes.c_int(X.shape[1]),  # nombre de lignes dans les données d'entrée
                        ctypes.c_int(X.shape[2]),  # nombre de colonnes dans les données d'entrée
                        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  # données de sortie
                        ctypes.c_int(Y.shape[2]),  # nombre de colonnes dans les données de sortie
                        ctypes.c_float(0.1),  # alpha
                        ctypes.c_int(100000),  # nb_iter
                        ctypes.c_bool(True))  # is_classification

    output_ptr = lib.predict_mlp_model(model_ptr,
                                       X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                       # données d'entrée
                                       ctypes.c_int(X.shape[1]),
                                       ctypes.c_int(X.shape[2]),
                                       # nombre de colonnes dans les données d'entrée
                                       ctypes.c_bool(True))  # is_classification

    # Obtenir le nombre total d'éléments que output_ptr est censé pointer
    num_elements = X.shape[1]

    # Créez un type de tableau ctypes de la bonne taille
    ArrayType = ctypes.c_double * num_elements

    # Cast le pointeur en un tableau ctypes
    output_array = ctypes.cast(output_ptr, ctypes.POINTER(ArrayType)).contents

    # Convertir le tableau ctypes en un numpy array
    output_np_array = np.ctypeslib.as_array(output_array)

    weights = np.ctypeslib.as_array(model_ptr.weights, shape=(num_features * k,))
    bias = np.ctypeslib.as_array(model_ptr.bias, shape=(k,))

    print(output_np_array)

    X = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ])

    colors = ['#FFAAAA', '#AAAAFF']

    plot_classification(X, output_np_array, colors)

    plt.show()

    # Libérer la mémoire du modèle
    lib.delete_mlp_model(model_ptr)


test_train()
