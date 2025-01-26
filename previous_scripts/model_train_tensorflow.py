import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
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
dataset_path = "dataset/train"
img_size = 200  # Tamaño de las imágenes
batch_size = 128

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

train_generator, val_generator = utils_file.get_tf_datagen(dataset_path=dataset_path, img_width=img_size,img_height=img_size, batch_size=batch_size)

# Obtener el número de clases
num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())
print(f"Clases detectadas: {class_names}")

# Construir el modelo CNN
model = utils_file.get_tf_model(num_classes=num_classes)

# Compilar el modelo
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Entrenar el modelo
history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# Guardar el modelo entrenado
model_path = "modelos/"
model_name = "tensorflow_model"
model_number = 1
for root, dirs, files in os.walk(model_path):
    for file in files:
        if file.startswith(model_name):
            model_number += 1

model_path = os.path.join(model_path, f"{model_name}_{model_number}.h5")            
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
