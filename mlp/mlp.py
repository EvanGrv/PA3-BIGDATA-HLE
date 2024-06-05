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


'''def plot_classification(X, predictions, colors):
    # Tracé de la séparation des classes
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    step = 0.1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Tracé des points d'entraînement avec des couleurs différentes pour chaque classe
    if predictions.ndim == 1:
        class_0 = X[predictions < 0]  # Filtrer les exemples de classe 0
        class_1 = X[predictions > 0]  # Filtrer les exemples de classe 1
    else:
        class_0 = X[predictions[:, 0] < 0]  # Filtrer les exemples de classe 0
        class_1 = X[predictions[:, 0] > 0]  # Filtrer les exemples de classe 1

    # Tracer les points des deux classes
    plt.scatter(class_0[:, 0], class_0[:, 1], color=colors[0], edgecolor='k', label='Classe 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], color=colors[1], edgecolor='k', label='Classe 1')

    # Afficher le graphique
    plt.legend()
    plt.show()'''

def plot_classification(X, predictions, colors, classification):
    # Tracé de la séparation des classes
    if classification:
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
            # Tracé de la séparation des classes pour données 2D
            x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
            y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
            step = 0.1

            xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
            grid_points = np.c_[xx.ravel(), yy.ravel()]

            # Tracé des points d'entraînement avec des couleurs différentes pour chaque classe
            class_0 = X[predictions[:, 0] < 0]
            class_1 = X[predictions[:, 0] > 0]
            class_2 = X[np.argmax(predictions, axis=1) == 2]

            plt.scatter(class_0[:, 0], class_0[:, 1], color=colors[0], edgecolor='k', label='Classe 0')
            plt.scatter(class_1[:, 0], class_1[:, 1], color=colors[1], edgecolor='k', label='Classe 1')
            plt.scatter(class_2[:, 0], class_2[:, 1], color=colors[2], edgecolor='k', label='Classe 2')

            # Calcul de la ligne de séparation pour un modèle 2D
            x_vals = np.array(plt.gca().get_xlim())
            '''for i in range(k):
                y_vals = -(x_vals * weights[i * num_X] + bias[i]) / weights[i * num_X + 1]
                plt.plot(x_vals, y_vals, '--', c='black')'''

            plt.legend()
            plt.show()
    else:
        # Régression linéaire
        plt.scatter(X, predictions, color='blue', label='Data points')

        plt.show()

        if X.shape == 1:
            x_vals = np.linspace(X.min(), X.max(), 100)
            #y_vals = weights[0] * x_vals + bias[0]
            #plt.plot(x_vals, y_vals, color='red', label='Regression line')
        else:
            raise ValueError("Ce code ne gère actuellement que la régression linéaire pour une seule caractéristique.")


def test_train():

    # Cas de test Classification

    # Test 1
    '''X = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ], dtype=np.float64)

    Y = np.array([
        [1],
        [-1],
        [1]
    ], dtype=np.float64)

    arr = [2, 1]'''

    # test 2
    '''X = np.concatenate([np.random.random((50,2)) * 0.9 + np.array([1, 1]), np.random.random((50,2)) * 0.9 + np.array([2, 2])], dtype=np.float64)
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0], dtype=np.float64)
    
    arr = [2, 1]
    '''

    # test 3
    '''X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=np.float64)
    Y = np.array([[1, 1, -1, -1]], dtype=np.float64)
    
    arr = [2, 5, 1]'''


    # Test 4
    '''X = np.random.random((500, 2)).astype(np.float64) * 2.0 - 1.0
    Y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X], dtype=np.float64)
    
    arr = [2, 4, 1]
    '''


    '''X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1, -1, -1] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                        [-1, 1, -1] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
                        [-1, -1, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
                        [0, 0, 0] for p in X], dtype=np.float64)

    X = X[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    Y = Y[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    
    arr = [2, 3]
    '''

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
    classification = False

    print(f"X shape : {X.shape}")
    print(f"Y chape : {Y.shape}")

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

    colors = {0: 'red', 1: 'blue', 2: 'green'}

    plot_classification(X, output_np_array, colors, classification)

    # Libérer la mémoire du modèle
    lib.delete_mlp_model(model_ptr)


test_train()
