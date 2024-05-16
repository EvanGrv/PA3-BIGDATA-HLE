import numpy as np
import ctypes
from matplotlib import pyplot as plt

# Chargement de la bibliothèque C
lib = ctypes.CDLL("target/debug/linear_model.dll")


# Définition de la structure LinearModel en Python
class LinearModel(ctypes.Structure):
    _fields_ = [
        ("weights", ctypes.POINTER(ctypes.c_double)),
        ("bias", ctypes.POINTER(ctypes.c_double)),
        ("loss", ctypes.POINTER(ctypes.c_double)),
        ("loss_size", ctypes.c_ulong),
    ]

    # Définition de la signature de la fonction train_linear_model


lib.train_linear_model.restype = LinearModel
lib.train_linear_model.argtypes = [
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

    # Tracé de la séparation des classes
    x_min, x_max = features[:, 0].min() - 0.1, features[:, 0].max() + 0.1
    y_min, y_max = features[:, 1].min() - 0.1, features[:, 1].max() + 0.1
    step = 0.1

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

    # Tracé des points d'entraînement avec des couleurs différentes pour chaque classe
    class_0 = features[outputs[:, 0] < 0]
    class_1 = features[outputs[:, 0] > 0]

    plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', edgecolor='k', label='Classe 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], color='red', edgecolor='k', label='Classe 1')

    # Calcul de la ligne de séparation
    if num_features == 2:
        x_vals = np.array(plt.gca().get_xlim())
        y_vals = -(x_vals * weights[0] + bias) / weights[1]
        plt.plot(x_vals, y_vals, '--', c='black')

    # Tracé de la séparation des classes avec des couleurs de fond différentes
    z = grid_predictions.reshape(xx.shape)
    plt.contourf(xx, yy, z, levels=[-np.inf, 0, np.inf], colors=['#1E3F66', '#8B0000'], alpha=0.5)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.show()


# Exemple de Classification
# test 1
inputs = np.array([[1, 1], [2, 3], [3, 3]], dtype=np.float64)
outputs = np.array([[1], [-1], [-1]], dtype=np.float64)

# test 2
# inputs = np.concatenate([np.random.random((50,2)) * 0.9 + np.array([1, 1]), np.random.random((50,2)) * 0.9 + np.array([2, 2])])
# outputs = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

# test 3
# inputs = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
# outputs = np.array([1, 1, -1, -1])

# test 4
# inputs = np.random.random((500, 2)) * 2.0 - 1.0
# outputs = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X])

# test 5
# inputs = np.random.random((500, 2)) * 2.0 - 1.0
# outputs = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X])

# test 6
# inputs = np.random.random((500, 2)) * 2.0 - 1.0
# outputs = np.array([[1, -1, -1] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
#              [-1, 1, -1] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else              [-1, -1, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
#              [0, 0, 0]for p in inputs], dtype=np.float64)

# inputs = inputs[[not np.all(arr == [0, 0, 0]) for arr in outputs]]
# outputs = outputs[[not np.all(arr == [0, 0, 0]) for arr in outputs]]

# test 7
# inputs = np.random.random((1000, 2)) * 2.0 - 1.0
# outputs = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [0, 1, 0] if abs(p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [0, 0, 1] for p in X])


# Exemple de Régression
# Test 1
'''inputs = np.array([
      [1.0],
      [2.0]
])
outputs = np.array([
      2.0,
      3.0
])'''

num_samples, num_features = inputs.shape
learning_rate = 0.01
num_iterations = 100000
k = 1

test(inputs, outputs, num_samples, num_features, learning_rate, num_iterations, k, True)
