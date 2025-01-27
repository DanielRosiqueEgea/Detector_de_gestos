import cv2 as cv
import mediapipe as mp
import os
import numpy as np

import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils_file import *

import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split


def main(model_to_test = "modelos/model_vgg16.h5",dataset_path = "dataset/train"):
    

    class_indices, index_to_class = get_class_indices()

    # Cargar imágenes y etiquetas
    x_images, y_labels = load_images(directory=dataset_path, uniq_labels=class_indices.keys(), progbar=True)

    print("Total number of Classes:", len(class_indices))
    print("Number of training images:", len(x_images))


    # Convertir etiquetas a one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_labels, num_classes=len(class_indices))

    # Cargar el modelo preentrenado

    model = load_model(model_to_test)

    # Evaluación del modelo
    y_pred = model.predict(x_images)
    
    # Generar matriz de confusión
    plot_confusion_matrix(y_train, y_pred, class_indices)
    
    calculate_metrics(class_indices, y_train, y_pred, "resultados/resultados_propio.txt")


def test_one_image(model_to_test = "modelos/model_vgg16.h5",test_path = "dataset/test/A_test.jpg"):   
    # Cargar y preprocesar imagen de prueba
    test_image = cv.imread(test_path)
    test_image = cv.resize(test_image, (64, 64))
    test_image = test_image / 255.0  # Normalización
    test_image = np.expand_dims(test_image, axis=0)  # Ajuste de dimensiones

    model = load_model(model_to_test)

    y_pred = model.predict(test_image)
    y_pred = np.argmax(y_pred, axis=1)[0]  # Extraer el índice de la clase predicha

    class_indices, index_to_class = get_class_indices()

    predicted_class = index_to_class[y_pred]  # Obtener el nombre de la clase
    print(predicted_class)

if __name__ == "__main__":
    # main()
    dataset_path = "dataset_propio"
    test_path = f"{dataset_path}/test"
    model_name = "vgg16"
    model_to_test = f"modelos/modelo_propio_callbacks_1.h5"
    test_image = f"{test_path}/A/A_602.jpg"

    test_one_image(model_to_test,test_image)
    
    main(model_to_test,test_path)