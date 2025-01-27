import os
import shutil
import random

# Definir directorios
dataset_dir = "dataset_propio"
train_dir = f"{dataset_dir}/train"
test_dir = f"{dataset_dir}/test"

# Crear el directorio test si no existe
os.makedirs(test_dir, exist_ok=True)

# Iterar sobre cada subdirectorio en train/
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)

    # Verificar si es un directorio
    if os.path.isdir(class_path):
        # Crear la misma estructura dentro de test/
        test_class_path = os.path.join(test_dir, class_name)
        os.makedirs(test_class_path, exist_ok=True)

        # Listar imágenes en el subdirectorio
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

        # Seleccionar 10% de imágenes al azar
        num_test_images = max(1, int(len(images) * 0.1))  # Asegurar al menos 1 imagen
        test_images = random.sample(images, num_test_images)

        # Mover las imágenes seleccionadas a test/
        for image in test_images:
            src_path = os.path.join(class_path, image)
            dest_path = os.path.join(test_class_path, image)
            shutil.move(src_path, dest_path)
            print(f"Movido: {src_path} → {dest_path}")

print("Separación de imágenes completada.")
