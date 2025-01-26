import cv2 as cv
import mediapipe as mp
import os
import numpy as np
import itertools
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils_file import *
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

def plot_confusion_matrix(y, y_pred, class_indices):
    """Genera y muestra la matriz de confusión con etiquetas correctas."""
    y = np.argmax(y, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(10, 8))
    ax = plt.subplot()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
    plt.colorbar()
    plt.title("Confusion Matrix")
    
    labels = list(class_indices.keys())  # Se obtiene la lista de etiquetas de clases
    tick_marks = np.arange(len(labels))
    
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Anotaciones dentro de la matriz
    limit = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                 color="white" if cm[i, j] > limit else "black")
    
    plt.show()

def main(model_to_test = "modelos/model_vgg16.h5",train_dir = "dataset/train"):
    

    class_indices, index_to_class = get_class_indices()

    # Cargar imágenes y etiquetas
    x_images, y_labels = load_images(directory=train_dir, uniq_labels=class_indices.keys(), progbar=True)


    print("Total number of gestures:", len(class_indices))
    print("Number of training images:", len(x_images))


    # Convertir etiquetas a one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_labels, num_classes=len(class_indices))

    # Mostrar ejemplos de entrenamiento
    # mostrar_ejemplos_matplotlib(x_images, y_train, index_to_class)

    # Cargar el modelo preentrenado

    model = load_model(model_to_test)

    # Evaluación del modelo
    y_pred = model.predict(x_images)
    
    # Generar matriz de confusión
    plot_confusion_matrix(y_train, y_pred, class_indices)

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
    
    model_to_test = "modelos/model_vgg16_with_my_data.h5"
    test_image = "test_accuracy/A/A_211.jpg"
    test_dataset = "test_accuracy"
    test_one_image(model_to_test,test_image)
    
    main(model_to_test,test_dataset)