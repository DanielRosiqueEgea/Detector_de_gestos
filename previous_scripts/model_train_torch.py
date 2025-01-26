import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import random
import cv2
import numpy as np
import time
from tqdm import tqdm

#  Configuración de dispositivo: GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando:", device)

#  Transformaciones para preprocesar las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#  Cargar el dataset ASL
train_dir = "dataset/train"
test_dir = "dataset/test"

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
# test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Obtener índices de imágenes reducidas
class_indices = {}
print("obteniendo datos reducidos")
for idx, (img, label) in tqdm(enumerate(train_dataset), total=len(train_dataset), desc="Procesando imágenes"):
    if label not in class_indices:
        class_indices[label] = []
    class_indices[label].append(idx)
# Seleccionar solo el 30% de imágenes por clase
selected_indices = []

for indices in tqdm(class_indices.values(), total=len(class_indices), desc="Reduciendo dataset"):
    selected_indices.extend(random.sample(indices, int(len(indices) * 0.3)))

# Crear dataset reducido
print(len(selected_indices))
print("dataset reducido")
reduced_dataset = Subset(train_dataset, selected_indices)


train_loader = DataLoader(reduced_dataset, batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

#  Obtener clases
classes = train_dataset.classes
num_classes = len(classes)
print("Clases:", classes)

#  Modelo: ResNet18 preentrenado
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Ajustar la capa de salida
model = model.to(device)

#  Función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#  Entrenamiento
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / len(train_loader))

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Guardar modelo entrenado
torch.save(model.state_dict(), "asl_model.pth")
print("Modelo guardado como asl_model.pth")

#  Función para predecir imágenes
def predict(image, model):
    model.eval()
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

#  Captura de video en tiempo real y clasificación
cap = cv2.VideoCapture(0)
time.sleep(2)  # Espera para encender la cámara

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Procesar imagen
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    prediction = predict(img, model)

    # Mostrar resultado
    cv2.putText(frame, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Presiona ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
