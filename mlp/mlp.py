import ctypes
import numpy as np
import matplotlib.pyplot as plt
import json
from mpl_toolkits.mplot3d import Axes3D

lib = ctypes.CDLL("target/release/mlp.dll")

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
    ctypes.c_bool,  # Modifier le type ici
    ctypes.c_size_t,
    ctypes.c_double,
]

lib.load_mlp.restype = ctypes.c_void_p  # Remplacez `ctypes.c_void_p` par le type de retour correct
lib.load_mlp.argtypes = [ctypes.c_char_p]

# Définition de la signature de la fonction create_mlp
lib.create_mlp.argtypes = (ctypes.POINTER(ctypes.c_uint), ctypes.c_size_t)
lib.create_mlp.restype = ctypes.c_void_p

def lire_fichier_json(nom_fichier):
    with open(nom_fichier, 'r') as fichier:
        contenu = json.load(fichier)
    return contenu


def test(inputs, outputs, n, title, k, iteration_count, alpha,isclassification ):

    # Définition des paramètres
    npl = np.array(n)

    # Conversion du tableau npl en tableau de type ctypes.c_uint
    npl_array = (ctypes.c_uint * len(npl))(*npl)

    # Création du MLP
    mlp = lib.create_mlp(npl_array, len(npl))

    # Entraînement du MLP
    lib.train_mlp(mlp, inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(inputs),
                  outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(outputs),
                  ctypes.c_bool(isclassification), iteration_count, alpha)

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

    if isclassification:
        if num_feature == 3:
            # Tracé de la séparation des classes pour données 3D
            x_min, x_max = inputs[:, 0].min() - 0.1, inputs[:, 0].max() + 0.1
            y_min, y_max = inputs[:, 1].min() - 0.1, inputs[:, 1].max() + 0.1
            z_min, z_max = inputs[:, 2].min() - 0.1, inputs[:, 2].max() + 0.1
            step = 0.1

            xx, yy, zz = np.meshgrid(
                np.arange(x_min, x_max, step),
                np.arange(y_min, y_max, step),
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
            x_min, x_max = inputs[:, 0].min() - 0.1, inputs[:, 0].max() + 0.1
            y_min, y_max = inputs[:, 1].min() - 0.1, inputs[:, 1].max() + 0.1
            step = 0.01
            xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            if k <= 2:
                for i in range(len(grid_points)):
                    input_ptr = grid_points[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                    output_ptr = lib.mlp_predict(mlp, input_ptr, len(grid_points[i]), ctypes.c_bool(isclassification))
                    predicted_output = np.array([output_ptr[j] for j in range(npl[-1])])
                    lib.mlp_free(output_ptr)
                    predicted_outputs.append(predicted_output)

                    #print("Input:", inputs[i], "Predicted output:", predicted_output, "resulat", outputs[i])

                # Conversion des prédictions en couleurs pour le tracé du graphe
                plt.scatter(inputs[:, 0], inputs[:, 1], c=np.argmax(outputs, axis=1))
                predicted_outputs = np.array(predicted_outputs)
                contour = predicted_outputs[:, 0].reshape(xx.shape)

                plt.contourf(xx, yy, contour, levels=[-np.inf, 0, np.inf], colors=['blue', 'red'], alpha=0.5)
                plt.title(title)
                # Affichage du graphe
                plt.show()
            else:
                grid_predictions = np.zeros((len(grid_points), k))  # Update this line

                for i in range(len(grid_points)):
                    input_ptr = grid_points[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                    output_ptr = lib.mlp_predict(mlp, input_ptr, len(grid_points[i]),
                                                 ctypes.c_bool(isclassification))
                    predicted_output = np.array([output_ptr[j] for j in range(npl[-1])])
                    lib.mlp_free(output_ptr)
                    grid_predictions[i] = np.ctypeslib.as_array(predicted_output, shape=(k,))

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

        predicted_outputs = []

        fichier_json = "./mlp_model.json"

        # Convertir les listes en tableaux numpy avec dtype=object

        mlp_from_file = lib.load_mlp(fichier_json.encode('utf-8'))

        for i in range(len(inputs)):
            input_ptr = inputs[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            output_ptr = lib.mlp_predict(mlp_from_file, input_ptr, len(inputs[i]), ctypes.c_bool(isclassification))
            predicted_output = np.array([output_ptr[j] for j in range(npl[-1])])
            lib.mlp_free(output_ptr)
            predicted_outputs.append(predicted_output)

            print("Input:", inputs[i], "Predicted output:", predicted_output, "resulat", outputs[i])

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(inputs[:, 0], inputs[:, 1], predicted_output)
        plt.show()
        plt.clf()

    lib.mlp_free(mlp)


def linear_simple():
    # test1 Linear Simple

    X = np.array([[1, 1], [2, 3], [3, 3]], dtype=np.float64)
    Y = np.array([[1], [-1], [-1]], dtype=np.float64)

    arr = [2, 1]

    alpha = 0.01
    nb_iter = 100000
    is_classification = True

    test(X, Y, arr, "test1", 2, nb_iter, alpha, is_classification)


def linear_multiple():
    # test2 Linear Multiple

    X = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    arr = [2, 1]

    alpha = 0.01
    nb_iter = 100000
    is_classification = True

    test(X, Y, arr, "test2", 2, nb_iter, alpha, is_classification)


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

    test(X, Y, arr, "test3", 2, nb_iter, alpha, is_classification)


def cross():
    # test4 Cross

    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1] if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else [-1] for p in X], dtype=np.float64)

    arr = [2, 4, 1]

    alpha = 0.01
    nb_iter = 1000000
    is_classification = True

    test(X, Y, arr, "test4", 2, nb_iter, alpha, is_classification)


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

    test(X, Y, arr, "test5", 3, nb_iter, alpha, is_classification)


def multi_cross():
    # test6 Multi Cross

    X = np.random.random((1000, 2)) * 2.0 - 1.0
    Y = np.array([[1, -1, -1] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [-1, 1, -1] if abs(
        p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [-1, -1, 1] for p in X])
    Y = np.array(Y, dtype=np.float64)

    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)

    arr = [2, 4, 4, 3]

    alpha = 0.001
    nb_iter = 1000000
    is_classification = True

    test(X, Y, arr, "test6", 3, nb_iter, alpha, is_classification)


def cas_de_test():
    #linear_simple()
    #linear_multiple()
    #xor()
    #cross()
    #multi_linear_classes()
    multi_cross()


cas_de_test()
