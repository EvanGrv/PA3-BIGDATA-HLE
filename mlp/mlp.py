import numpy as np
import ctypes
from matplotlib import pyplot as plt

lib = ctypes.CDLL("..\\mlp\\target\\debug\\mlp.dll")


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
lib.create_mlp_model.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t]

# Fonction train_mlp_model
lib.train_mlp_model.argtypes = [ctypes.POINTER(MLP),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.c_int64,
                                ctypes.c_int64,
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.c_int64,
                                ctypes.c_double,
                                ctypes.c_int64,
                                ctypes.c_bool]

# Fonction predict_mlp_model
lib.predict_mlp_model.restype = ctypes.POINTER(ctypes.c_double)
lib.predict_mlp_model.argtypes = [ctypes.POINTER(MLP),
                                  ctypes.POINTER(ctypes.c_double),
                                  ctypes.c_int64,
                                  ctypes.c_int64,
                                  ctypes.c_bool]

# Fonction delete_mlp_model
lib.delete_mlp_model.argtypes = [ctypes.POINTER(MLP)]


def plot_classification(X, predictions, classification):
    # Tracé de la séparation des classes
    if classification:
        k = len(predictions[0])

        if X.shape == 3:
            # Tracé de la séparation des classes pour données 3D
            x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
            y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
            z_min, z_max = X[:, 2].min() - 0.1, X[:, 2].max() + 0.1
            step = 0.1

            xx, yy, zz = np.meshgrid(
                np.arange(x_min, x_max, step),
                np.arange(y_min, y_max, step),
                np.arange(z_min, z_max, step)
            )
            grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

            # Tracé des points d'entraînement avec des couleurs différentes pour chaque classe
            class_0 = X[predictions[:, 0] < 0]
            class_1 = X[predictions[:, 0] > 0]
            class_2 = X[np.argmax(predictions, axis=1) == 2]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(class_0[:, 0], class_0[:, 1], class_0[:, 2], color='blue', edgecolor='k', label='Classe 0')
            ax.scatter(class_1[:, 0], class_1[:, 1], class_1[:, 2], color='red', edgecolor='k', label='Classe 1')
            ax.scatter(class_2[:, 0], class_2[:, 1], class_2[:, 2], color='green', edgecolor='k', label='Classe 2')

            # Calcul de la ligne de séparation pour un modèle 3D
            x_vals = np.array([x_min, x_max])
            y_vals = np.array([y_min, y_max])
            '''for i in range(k):
                xx, yy = np.meshgrid(x_vals, y_vals)
                zz = -(weights[i * num_X] * xx + weights[i * num_X + 1] * yy + bias[i]) / weights[
                    i * num_X + 2]
                ax.plot_surface(xx, yy, zz, alpha=0.5)'''

            plt.legend()
            plt.show()

        elif X.shape == 2:
            # Prédire les sorties pour une grille de points
            x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
            y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
            step = 0.01
            xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            grid_predictions = np.zeros((len(grid_points), len(predictions[0])))

            # Tracé des points d'entraînement avec des couleurs différentes pour chaque classe
            class_0 = X[predictions[:, 0] < 0]
            class_1 = X[predictions[:, 0] > 0]

            plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', edgecolor='k', label='Classe 0')
            plt.scatter(class_1[:, 0], class_1[:, 1], color='red', edgecolor='k', label='Classe 1')

            # Tracer la séparation des classes
            contour = grid_predictions[:, 0].reshape(xx.shape)
            plt.contourf(xx, yy, contour, levels=[-np.inf, 0, np.inf], colors=['blue', 'red'], alpha=0.5)

    else:
        # Régression linéaire
        plt.scatter(X, predictions, color='blue', label='Data points')

        plt.show()

        if X.shape == 1:
            x_vals = np.linspace(X.min(), X.max(), 100)
            # y_vals = weights[0] * x_vals + bias[0]
            # plt.plot(x_vals, y_vals, color='red', label='Regression line')
        else:
            raise ValueError("Ce code ne gère actuellement que la régression linéaire pour une seule caractéristique.")

    plt.show()

def plot_classification2(X, weights, bias, colors):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    # Générer une grille de points couvrant tout l'espace 2D
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Calculer les valeurs de décision
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], weights) + bias
    Z = (Z > 0).astype(int)
    Z = Z.reshape(xx.shape)

    # Tracer les points de données avec les couleurs des classes prédites
    plt.scatter(X[:, 0], X[:, 1], c=[colors[int(p > 0)] for p in np.dot(X, weights) + bias], edgecolor='k')

    # Tracer la ligne de séparation
    plt.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=[2])

    # Afficher le graphique
    plt.show()


def show_test(X_train, Y_train, predictions):
    # Prédire les sorties pour une grille de points
    x_min, x_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
    y_min, y_max = X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1
    step = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    # Assurez-vous que predictions est 1D si nécessaire
    if len(predictions.shape) > 1 and predictions.shape[1] == 1:
        predictions = predictions.ravel()

    # Vérifiez et reshaper predictions
    if predictions.size != xx.size:
        raise ValueError(f"Size of predictions ({predictions.size}) does not match the size of the grid ({xx.size})")

    grid_predictions = predictions
    k = Y_train.shape[1]
    if k <= 2:
        # Tracé des points d'entraînement avec des couleurs différentes pour chaque classe
        class_0 = X_train[Y_train[:, 0] < 0]
        class_1 = X_train[Y_train[:, 0] > 0]

        plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', edgecolor='k', label='Classe 0')
        plt.scatter(class_1[:, 0], class_1[:, 1], color='red', edgecolor='k', label='Classe 1')

        # Tracer la séparation des classes
        contour = grid_predictions.reshape(xx.shape)
        plt.contourf(xx, yy, contour, levels=[-np.inf, 0, np.inf], colors=['blue', 'red'], alpha=0.5)

    else:
        # Tracé des points d'entraînement avec des couleurs différentes pour chaque classe
        class_0 = X_train[Y_train[:, 0] < 0]
        class_1 = X_train[Y_train[:, 0] > 0]
        class_2 = X_train[np.argmax(Y_train, axis=1) == 2]

        plt.scatter(class_0[:, 0], class_0[:, 1], color='red', edgecolor='k', label='Classe 0')
        plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', edgecolor='k', label='Classe 1')
        plt.scatter(class_2[:, 0], class_2[:, 1], color='green', edgecolor='k', label='Classe 2')

        # Tracer la séparation des classes
        contour = np.argmax(grid_predictions, axis=1).reshape(xx.shape)
        plt.contourf(xx, yy, contour, levels=[-np.inf, 0.5, 1.5, np.inf], colors=['blue', 'red', 'green'], alpha=0.4)

    plt.show()


def test(X, Y, arr, learning_rate, nb_iter, classification):
    arr_ptr = (ctypes.c_longlong * len(arr))(*arr)
    model_ptr = lib.create_mlp_model(arr_ptr, len(arr))

    # Appeler la fonction train_mlp_model pour entraîner le modèle
    lib.train_mlp_model(model_ptr,
                        X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  # données d'entrée
                        ctypes.c_int64(X.shape[0]),  # nombre de lignes dans les données d'entrée
                        ctypes.c_int64(X.shape[1]),  # nombre de colonnes dans les données d'entrée
                        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  # données de sortie
                        ctypes.c_int64(Y.shape[1]),  # nombre de colonnes dans les données de sortie
                        ctypes.c_double(learning_rate),  # alpha
                        ctypes.c_int64(nb_iter),  # nb_iter
                        ctypes.c_bool(classification))  # is_classification

    output_ptr = lib.predict_mlp_model(model_ptr,
                                       X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                       # données d'entrée
                                       ctypes.c_int64(X.shape[0]),
                                       ctypes.c_int64(X.shape[1]),
                                       # nombre de colonnes dans les données d'entrée
                                       ctypes.c_bool(classification))  # is_classification

    # Obtenir le nombre total d'éléments que output_ptr est censé pointer
    num_elements = X.shape[0]

    # Créez un type de tableau ctypes de la bonne taille
    ArrayType = ctypes.c_double * num_elements

    # Cast le pointeur en un tableau ctypes
    output_array = ctypes.cast(output_ptr, ctypes.POINTER(ArrayType)).contents

    # Convertir le tableau ctypes en un numpy array
    output_np_array = np.ctypeslib.as_array(output_array)

    print(output_np_array)

    #show_test(X, Y, output_np_array)

    plot_classification2(X, weights, bias, colors)

    # Libérer la mémoire du modèle
    lib.delete_mlp_model(model_ptr)


def test_train():
    # Cas de test Classification

    # Test 1
    # X = np.array([
    #     [1, 1],
    #     [2, 3],
    #     [3, 3]
    # ], dtype=np.float64)
    #
    # Y = np.array([
    #     [1],
    #     [-1],
    #     [-1]
    # ], dtype=np.float64)
    #
    # arr = [2, 1]

    # test 2
    '''X = np.concatenate([np.random.random((50,2)) * 0.9 + np.array([1, 1]), np.random.random((50,2)) * 0.9 + np.array([2, 2])], dtype=np.float64)
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0], dtype=np.float64)
    
    arr = [2, 1]
    '''

    # test 3 Xor
    '''X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=np.float64)
    Y = np.array([[1, 1, -1, -1]], dtype=np.float64)
    
    arr = [2, 2, 1]'''

    # Test 4 Cross
    '''X = np.random.random((500, 2)).astype(np.float64) * 2.0 - 1.0
    Y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X], dtype=np.float64)
    
    arr = [2, 4, 1]
    '''

    # Test 5 multi class
    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1, -1, -1] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                        [-1, 1, -1] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
                        [-1, -1, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
                        [0, 0, 0] for p in X], dtype=np.float64)

    X = X[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    Y = Y[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    
    arr = [2, 3]

    # Test 6 multi Cross
    '''X = np.random.random((1000, 2)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [0, 1, 0] if abs(
        p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [0, 0, 1] for p in X])

    arr = [2, 1, 1, 3]'''

    # Cas de test Régression

    '''X = np.array([
        [1],
        [2]
    ])
    Y = np.array([
        [2, 3]
    ])

    arr = [1, 1]'''

    learning_rate = 0.01
    nb_iter = 10000
    classification = True

    print(f"X shape : {X.shape}")
    print(f"Y chape : {Y.shape}")

    test(X, Y, arr, learning_rate, nb_iter, classification)


test_train()
