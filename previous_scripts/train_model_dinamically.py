import cv2 as cv
import mediapipe as mp
import os
import numpy as np
from pynput import keyboard, mouse
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# ---- Configuración General ----
exit_program = False
capture_hand = False
base_dir = "test_accuracy"
testing_char = "del"
output_dir = f"{base_dir}/{testing_char}"
os.makedirs(output_dir, exist_ok=True)

# ---- Diccionarios de Clases ----
class_indices = {chr(i + 65): i for i in range(26)}  # 'A' a 'Z'
class_indices.update({"del": 26, "nothing": 27, "space": 28})
index_to_class = {v: k for k, v in class_indices.items()}  # Invertir

# ---- Manejo de Teclado ----
def on_press(key):
    global exit_program, capture_hand, output_dir, testing_char
    try:
        if key == keyboard.Key.ctrl:
            exit_program = True
            return
        elif key == keyboard.Key.alt:
            capture_hand = True
            return
        elif "0" == key.char:
            testing_char = "del"
        else:
            testing_char = key.char.upper()
            
        output_dir = f"{base_dir}/{testing_char}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Capturando imágenes para: {testing_char}")
    except AttributeError:
        pass

def on_release(key):
    global capture_hand
    if key == keyboard.Key.alt:
        capture_hand = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# ---- Captura de Video y Mediapipe ----
mp_hands = mp.solutions.hands
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("⛔ No se pudo abrir la cámara")
    exit()

# ---- Cargar el Modelo ----
model_to_test = "modelos/tensorflow_model.h5"
model = load_model(model_to_test)
img_size = 200
frame_idx = 0
capture_delay = 0.1
last_capture_time = 0



# ---- Captura en Tiempo Real ----
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while cv.getWindowProperty("preview", cv.WND_PROP_VISIBLE) == 1:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_min = max(0, int(min([lm.x for lm in hand_landmarks.landmark]) * w))
                y_min = max(0, int(min([lm.y for lm in hand_landmarks.landmark]) * h))
                x_max = min(w, int(max([lm.x for lm in hand_landmarks.landmark]) * w))
                y_max = min(h, int(max([lm.y for lm in hand_landmarks.landmark]) * h))

                cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                current_time = time.time()

                if capture_hand and (current_time - last_capture_time) >= capture_delay:
                    hand_crop = frame[y_min:y_max, x_min:x_max]
                    hand_crop = cv.resize(hand_crop, (img_size, img_size))

                    # Guardar imagen en el dataset
                    crop_path = os.path.join(output_dir, f"{testing_char}_{frame_idx}.jpg")
                    cv.imwrite(crop_path, hand_crop)
                    print(f"Imagen guardada: {crop_path}")

                    last_capture_time = current_time
                    frame_idx += 1

        cv.imshow("preview", frame)
        if exit_program or (cv.waitKey(1) & 0xFF == 27):
            break

cap.release()
cv.destroyAllWindows()
listener.stop()

# ---- Entrenar después de capturar nuevas imágenes ----
train_model()
