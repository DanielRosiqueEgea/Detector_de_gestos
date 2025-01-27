import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import os
import seaborn as sns
from tqdm import tqdm
import itertools



def get_model_file(model_path, model_name):
    model_number = 1
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.startswith(model_name):
                model_number += 1

    return f"{model_name}_{model_number}.h5"

def get_class_indices():
    class_indices = {chr(i + 65): i for i in range(26)}  # 'A' a 'Z'
    class_indices.update({"del": 26, "nothing": 27, "space": 28})
    index_to_class = {v: k for k, v in class_indices.items()}
    return class_indices, index_to_class

def get_uniq_labels(directory):
    uniq_labels = sorted(os.listdir(directory))
    return uniq_labels
#Helper function to load images from given directories
def load_images(directory,uniq_labels,progbar=False,img_size=64):
    images = []
    labels = []

    total_files = sum(len(os.listdir(os.path.join(directory, label))) for label in uniq_labels)
    progress = tqdm(total=total_files) if progbar  else False

    for idx, label in enumerate(uniq_labels):
        for file in os.listdir(directory + "/" + label):
            filepath = directory + "/" + label + "/" + file
            image = cv.resize(cv.imread(filepath), (img_size, img_size))
            images.append(image)
            labels.append(idx)
            if progress:
                progress.update(1)
    if progress:
        progress.close()
    images = np.array(images)
    labels = np.array(labels)
    return(images, labels)



def get_tf_model_by_size(img_size=200):
    return get_tf_model(img_width=img_size, img_height=img_size)

def get_tf_model(img_width=200, img_height=200, num_classes=29):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu",
                      input_shape=(img_width, img_height, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model


def get_tf_datagen(dataset_path,img_width=200, img_height=200, batch_size=32):
    # Generador de datos con aumento (data augmentation)
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2,  # 20% para validación
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Cargar datos de entrenamiento y validación
    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    val_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )
    return train_generator, val_generator

def train_tf_model(model, model_path,train_generator, val_generator, epochs=10, learning_rate=0.0001):

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(train_generator, validation_data=val_generator, epochs=epochs)

    model.save(model_path, save_format="h5")
    print("Modelo actualizado y guardado.")
    return model, history



def preprocess_hand_crop_tf(frame, x_min, y_min, x_max, y_max, img_size):
    """Recorta y preprocesa la imagen de la mano."""
    hand_crop = frame[y_min:y_max, x_min:x_max]
    hand_crop = cv.resize(hand_crop, (img_size, img_size))
    hand_crop = hand_crop.astype("float32") / 255.0  # Normalizar
    hand_crop = np.expand_dims(hand_crop, axis=0)  # Añadir batch dimension
    return hand_crop


def predict_class_tf(model, hand_crop, index_to_class):
    """Realiza la predicción sobre la imagen de la mano."""
    prediction = model.predict(hand_crop)
    predicted_index = np.argmax(prediction)
    predicted_class = index_to_class[predicted_index]
    confidence = np.max(prediction)
    return predicted_class, confidence


def store_training_data(X_train, y_train, hand_crop, class_indices, testing_char):
    """Almacena la imagen en X_train y la etiqueta en y_train."""
    X_train.append(hand_crop[0])
    y_train.append(class_indices[testing_char])


def draw_prediction(frame, x_min, y_min, predicted_class, confidence):
    """Dibuja la predicción sobre el frame."""
    text = f"{predicted_class} ({confidence:.2f})"
    cv.putText(frame, text, (x_min, y_min - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return text


def save_cropped_image(output_dir, predicted_class, frame_idx, hand_crop):
    """Guarda la imagen procesada en la carpeta de salida."""
    if not isinstance(hand_crop, np.ndarray):
    #     print(f"Dimensiones de la imagen a guardar: {hand_crop.shape}")
    # else:
    #     print("ERROR: La imagen no es un array numpy")
        return
    if hand_crop is None or hand_crop.size == 0:
        # print("ERROR: La imagen recortada está vacía")
        return

    crop_path = os.path.join(output_dir, f"{predicted_class}_{frame_idx}.jpg")
    os.makedirs(output_dir, exist_ok=True)

    success = cv.imwrite(crop_path, hand_crop[0]*255.0)
    # if not success:
    #     print(f"Error al guardar la imagen en {crop_path}")
    
    
def show_hand_crop(hand_crop, window_name="hand_crop"):
    """Muestra la imagen de la mano."""
    hand_crop_display = (
        hand_crop[0] * 255).astype(np.uint8)  # Escalar a 0-255

    # Si la imagen es de un solo canal (grayscale), convertirla a 3 canales
    if len(hand_crop_display.shape) == 2:
        hand_crop_display = cv.cvtColor(hand_crop_display, cv.COLOR_GRAY2BGR)

    # Mostrar la imagen en otra ventana
    cv.imshow(window_name, hand_crop_display)



def mostrar_ejemplos_matplotlib(X_train, y_train, index_to_class, num_ejemplos=10):
    # Asegurar que no exceda el tamaño del dataset
    num_muestras = min(len(X_train), num_ejemplos)
    if num_muestras <= 1:
        print("No hay suficientes ejemplos para mostrar.")
        return
    fig, axes = plt.subplots(1, num_muestras, figsize=(15, 5))

    indeces = np.random.choice(len(X_train), num_muestras, replace=False)

    for i in range(num_muestras):
        idx = indeces[i]
        ax = axes[i]
        imagen = X_train[idx]
        if isinstance(y_train[idx], np.ndarray):
            label = np.argmax(y_train[idx])
        else:
            label = y_train[idx]
        etiqueta = index_to_class[label]  # Convertir índice a clase

        ax.imshow(imagen, cmap="gray")
        ax.set_title(etiqueta)
        ax.axis("off")  # Ocultar ejes

    plt.show()

def mostrar_ejemplos_opencv(X_train, y_train, index_to_class, num_ejemplos=10,  window_name="hand_crop"):
    # Asegurar que no exceda el tamaño del dataset
    num_muestras = min(len(X_train), num_ejemplos)
    if num_muestras <= 1:
        print("No hay suficientes ejemplos para mostrar.")
        return
    fig, axes = plt.subplots(1, num_muestras, figsize=(15, 5))

    indeces = np.random.choice(len(X_train), num_muestras, replace=False)

    for i in range(num_muestras):
        idx = indeces[i]
        ax = axes[i]
       
        imagen = X_train[idx]
        etiqueta = index_to_class[y_train[idx]]  # Convertir índice a clase

        ax.imshow(imagen, cmap="gray")
        ax.set_title(etiqueta)
        ax.axis("off")  # Ocultar ejes
    fig.canvas.draw()
    # Convertir a array de imagen
    img = np.array(fig.canvas.renderer.buffer_rgba())
    # Convertir de RGBA a BGR (formato OpenCV)
    img = cv.cvtColor(img, cv.COLOR_RGBA2BGR)

    # Mostrar con OpenCV
    cv.imshow(window_name, img)

def mostrar_matriz_confusion(cm,class_indices,cm_path = None):
 
    # Crear la figura
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_indices.keys(),
                yticklabels=class_indices.keys())
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")

    # Guardar la imagen de la matriz de confusión
    if cm_path is not None:
        plt.savefig(cm_path)
    plt.show()

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Entrenamiento")
    plt.plot(history.history["val_accuracy"], label="Validación")
    plt.title("Precisión del modelo")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Entrenamiento")
    plt.plot(history.history["val_loss"], label="Validación")
    plt.title("Pérdida del modelo")
    plt.legend()

    plt.show()


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



def calculate_metrics(class_indices, y_train, y_pred,output_report=None):
    y_true = np.argmax(y_train, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels, average='weighted')
    recall = recall_score(y_true, y_pred_labels, average='weighted')
    f1 = f1_score(y_true, y_pred_labels, average='weighted')

    report = classification_report(y_true, y_pred_labels, target_names=list(class_indices.keys()))
    print("\n===== Métricas de Evaluación =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:\n")
    print(report)

    if output_report is not None:
        with open(output_report, "w") as f:
            f.write(report)