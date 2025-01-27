import matplotlib.pyplot as plt
import numpy as np
from utils_file import *

def mostrar_ejemplos_matplotlib(X_train, y_train, index_to_class):
    clases_unicas = index_to_class.keys()
    num_clases = len(clases_unicas)
    num_columnas = 5  # Número de columnas en la cuadrícula
    num_filas = (num_clases // num_columnas) + (num_clases % num_columnas > 0)  # Calcular filas

    fig, axes = plt.subplots(num_filas, num_columnas, figsize=(num_columnas * 3, num_filas * 3))
    axes = axes.flatten()  # Convertir a una lista para iterar fácilmente

    for i, clase in enumerate(clases_unicas):
        indices = np.where(y_train == clase)[0]  # Obtener índices de la clase
        idx = np.random.choice(indices)  # Seleccionar un índice aleatorio de esa clase

        ax = axes[i]
        ax.imshow(X_train[idx], cmap="gray")
        ax.set_title(index_to_class[clase])  # Convertir índice a nombre de clase
        ax.axis("off")  # Ocultar ejes

    # Ocultar ejes de los espacios vacíos en la cuadrícula
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("mostrar_ejemplos_matplotlib.png")
    plt.show()




class_indices, index_to_class = get_class_indices()
dataset_path = "dataset_propio"
test_path = f"{dataset_path}/test"
# Cargar imágenes y etiquetas
x_images, y_labels = load_images(directory=test_path, uniq_labels=class_indices.keys(), progbar=True)

print("Total number of Classes:", len(class_indices))
print("Number of training images:", len(x_images))

mostrar_ejemplos_matplotlib(x_images, y_labels, index_to_class)