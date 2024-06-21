import numpy as np
import ctypes
import matplotlib.pyplot as plt


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


cas_de_test()
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
