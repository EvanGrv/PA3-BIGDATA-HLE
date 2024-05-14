import numpy as np
import ctypes
from matplotlib import pyplot as plt

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
                                  ctypes.c_bool]

# Fonction delete_mlp_model
lib.delete_mlp_model.argtypes = [ctypes.POINTER(MLP)]


def test_train():
    X = np.array([[
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]], dtype=np.float64)

    Y = np.array([[
        [-1.0],
        [1.0],
        [1.0],
        [-1.0]
    ]], dtype=np.float64)

    arr = [2, 2, 1]
    arr_ptr = (ctypes.c_longlong * len(arr))(*arr)
    model_ptr = lib.create_mlp_model(arr_ptr, len(arr))

    # Appeler la fonction train_mlp_model pour entraîner le modèle
    lib.train_mlp_model(model_ptr,
                        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  # données d'entrée
                        ctypes.c_int(X.shape[0]),  # nombre de lignes dans les données d'entrée
                        ctypes.c_int(X.shape[1]),  # nombre de colonnes dans les données d'entrée
                        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  # données de sortie
                        ctypes.c_int(Y.shape[1]),  # nombre de colonnes dans les données de sortie
                        ctypes.c_float(0.1),  # alpha
                        ctypes.c_int(100000),  # nb_iter
                        ctypes.c_bool(True))  # is_classification

    predicted_outputs_ptr = lib.predict_mlp_model(model_ptr,
                                                  X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                  # données d'entrée
                                                  ctypes.c_int(X.shape[1]),
                                                  # nombre de colonnes dans les données d'entrée
                                                  ctypes.c_bool(True))  # is_classification

    # Convertir le pointeur en tableau numpy
    predicted_outputs = np.ctypeslib.as_array(predicted_outputs_ptr, shape=(X.shape[0],))

    # Afficher les valeurs
    for value in predicted_outputs:
        print(value)

    # Afficher les prédictions
    print("Prédictions :", predicted_outputs)

    # Libérer la mémoire du modèle
    lib.delete_mlp_model(model_ptr)


test_train()
