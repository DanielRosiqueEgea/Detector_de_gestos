import cv2 as cv
import mediapipe as mp
import os
import numpy as np
from pynput import keyboard, mouse
import time
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from utils_file import *
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random

exit_program = False
capture_hand = False

base_dir = "test_accuracy"
testing_char = "del"
output_dir = f"{base_dir}/{testing_char}"

# Función para manejar eventos de teclado


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

expansion_factor = 0.06  # Valor inicial
# Función para manejar el expansion_factor
# Este valor controla el tamaño del bounding box
# Lo queremos para comprobar correcto funcionamiento del modelo


def on_scroll(x, y, dx, dy):
    global expansion_factor
    factor_change = 0.02
    if dy > 0:
        expansion_factor = min(0.5, expansion_factor +
                               factor_change)  # Aumenta el factor
    else:
        expansion_factor = max(0.01, expansion_factor -
                               factor_change)  # Reduce el factor
    print(f"Nuevo expansion_factor: {expansion_factor:.2f}")


mouse_listener = mouse.Listener(on_scroll=on_scroll)
mouse_listener.start()


# model_to_test = "modelos/model_vgg16_with_my_data.h5"
model_to_test = "modelos/modelo_propio_callbacks_1.h5"
model = load_model(model_to_test)

img_size = 64

class_indices, index_to_class = get_class_indices()

y_true = []  # Etiquetas reales
y_pred = []  # Predicciones
# Variables for training dynamically
X_train = []
y_train = []


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

for i in range(3):
    try:
        cap = cv.VideoCapture(3-i)
        if cap.isOpened():
            break
    except:
        pass

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

frame_idx = 0
capture_delay = 0.2  # Por ejemplo, 2 segundos de espera
last_capture_time = 0  # Inicialmente no hay capturas previas


cv.namedWindow("preview")
# cv.namedWindow("hand_crop")
os.makedirs(output_dir, exist_ok=True)



with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while cv.getWindowProperty('preview', cv.WND_PROP_VISIBLE) == 1:
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        frame_idx += 1
        if ret is False:
            print("No se pudo acceder a la cámara.")
            break

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if not results.multi_hand_landmarks:

            cv.imshow('preview', frame)
            if exit_program or (cv.waitKey(1) & 0xFF == 27):
                break
            continue

        for hand_landmarks in results.multi_hand_landmarks:

            # Calcular el bounding box de la mano
            h, w, _ = frame.shape
            # Calcular el bounding box expandido
            x_min = max(
                0, int((min([lm.x for lm in hand_landmarks.landmark]) - expansion_factor) * w))
            y_min = max(
                0, int((min([lm.y for lm in hand_landmarks.landmark]) - expansion_factor) * h))
            x_max = min(
                w, int((max([lm.x for lm in hand_landmarks.landmark]) + expansion_factor) * w))
            y_max = min(
                h, int((max([lm.y for lm in hand_landmarks.landmark]) + expansion_factor) * h))

            # Dibujar el bounding box
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            current_time = time.time()
            if (current_time - last_capture_time) >= capture_delay:
                hand_crop = preprocess_hand_crop_tf(
                    frame, x_min, y_min, x_max, y_max, img_size)

                # show_hand_crop(hand_crop)
                
                predicted_class, confidence = predict_class_tf(
                    model, hand_crop, index_to_class)

                y_true.append(testing_char)
                y_pred.append(predicted_class)

                # store_training_data(X_train, y_train, hand_crop,
                #                     class_indices, testing_char)
                
                # mostrar_ejemplos_opencv(X_train, y_train, index_to_class, num_ejemplos=10,  window_name="hand_crop")
                
                text = draw_prediction(
                    frame, x_min, y_min, predicted_class, confidence)
                # save_cropped_image(
                #     output_dir, testing_char, frame_idx, hand_crop)
        # text = f"{testing_char}"
        # Waits for a user input to quit the application
        cv.putText(frame, text, (x_min, y_min - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv.imshow("preview", frame)
        if exit_program or (cv.waitKey(1) & 0xFF == 27):
            break


cap.release()
cv.destroyAllWindows()

listener.stop()
mouse_listener.stop()
