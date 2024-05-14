import numpy as np
import ctypes
from matplotlib import pyplot as plt


def test(features, outputs, num_samples, num_features, learning_rate, num_iterations, k, classification):
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

    # Prédiction des sorties pour chaque exemple d'entraînement
    for i in range(num_samples):
        features_c = features[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        predictions_c = lib.predict_linear_model(
            features_c,
            linear_model.weights,
            linear_model.bias,
            num_samples,
            num_features,
            k,
            ctypes.c_bool(classification),
        )

        # Conversion du tableau C en tableau NumPy
        predictions = np.ctypeslib.as_array(predictions_c, shape=(k,))

        # Affichage des résultats
        print("Predictions:", predictions, "Target:", outputs[i])

    # Prédiction des sorties pour une grille de points

    # Tracé de la séparation des classes
    if k <= 2:
        x_min, x_max = -1., 1.
        y_min, y_max = -1., 1.
        step = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        # print("test ", grid_points[1][1])

        grid_predictions = np.array(grid_points)  # np.zeros((len(grid_points), len(outputs[)))
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
            grid_predictions[i] = np.ctypeslib.as_array(grid_predictions_c, shape=(k + 1,))
            # lib.free(grid_predictions_c)
            # print(grid_predictions)

        # Tracé des points d'entraînement
        plt.scatter(features[:, 0], features[:, 1], c=np.argmax(outputs, axis=1))

        contour = np.argmax(grid_predictions, axis=1).reshape(xx.shape)
        contour = grid_predictions[:, 0].reshape(xx.shape)
        plt.title("test4")

        plt.contourf(xx, yy, contour, levels=[-np.inf, 0, np.inf], colors=['blue', 'red'], alpha=0.5)
    else:
        x_min, x_max = -1., 1.
        y_min, y_max = -1., 1.
        step = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_predictions = np.zeros((len(grid_points), len(outputs[0])))
        print("test ", grid_points)
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
            # lib.free(grid_predictions_c)
            # print(grid_predictions)

        # Tracé des points d'entraînement
        plt.scatter(features[:, 0], features[:, 1], c=np.argmax(outputs, axis=1))
        contour = np.argmax(grid_predictions, axis=1).reshape(xx.shape)
        plt.contourf(xx, yy, contour, levels=[-np.inf, 0.5, 1.5, np.inf], colors=['blue', 'red', 'green'], alpha=0.5)

    # Affichage du tracé
    plt.show()


#test1

inputs = np.random.random((1000, 2)) * 2.0 - 1.0
outputs = np.array([[1, -1, -1] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [-1, 1, -1] if abs(p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [-1, -1, 1] for p in inputs])
outputs = np.array(outputs, dtype=np.float64)

num_samples, num_features = inputs.shape
learning_rate = 0.01
num_iterations = 100000
k = 1


test(inputs, outputs, num_samples, num_features, learning_rate, num_iterations, k, True)
