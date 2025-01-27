import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import utils_file

print("GPU Disponible:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU habilitada para TensorFlow")
    except RuntimeError as e:
        print(e)

# Definir rutas del dataset
dataset_path = "dataset_propio/train"
img_size = 64  # Tamaño de las imágenes
batch_size = 32

# Generador de datos con aumento (data augmentation)
datagen = ImageDataGenerator(
    rescale=1.0/255, 
    validation_split=0.1,  # 20% para validación
)

# Cargar datos de entrenamiento y validación
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# train_generator, val_generator = utils_file.get_tf_datagen(dataset_path=dataset_path, img_width=img_size,img_height=img_size, batch_size=batch_size)

# Obtener el número de clases
num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())
print(f"Clases detectadas: {class_names}")

# Construir el modelo CNN
model = utils_file.get_tf_model(img_height=img_size, img_width=img_size,num_classes=num_classes)

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),  # 🔹 Detiene si no mejora en 5 epochs
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-7),  # 🔹 Reduce LR si no mejora en 3 epochs
]

# Entrenar el modelo
print("Sentencia de entrenamiento:")
history = model.fit(train_generator, validation_data=val_generator, epochs=100, callbacks=callbacks, verbose=0)

# Guardar el modelo entrenado
model_path = "modelos/"
model_name = "modelo_propio_callbacks"


model_file = utils_file.get_model_file(model_path, model_name)

model_path = os.path.join(model_path, model_file)            
model.save(model_path)
print(f"Modelo guardado como {model_path}")

# Graficar precisión y pérdida
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

class_indices, index_to_class = utils_file.get_class_indices()

# Cargar imágenes y etiquetas
dataset_path = "dataset_propio/test"
x_images, y_labels = utils_file.load_images(directory=dataset_path, uniq_labels=class_indices.keys(), progbar=True)

print("Total number of Classes:", len(class_indices))
print("Number of training images:", len(x_images))


# Convertir etiquetas a one-hot encoding
y_train = tf.keras.utils.to_categorical(y_labels, num_classes=len(class_indices))

# Cargar el modelo preentrenado

# Evaluación del modelo
y_pred = model.predict(x_images)

# Generar matriz de confusión
utils_file.plot_confusion_matrix(y_train, y_pred, class_indices)

utils_file.calculate_metrics(class_indices, y_train, y_pred, "resultados/resultados_propio_callbacks.txt")
