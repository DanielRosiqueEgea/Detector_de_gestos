import cv2 as cv
import mediapipe as mp
import os
from pynput import keyboard, mouse
import time

exit_program = False
capture_hand = False

output_dir = "dataset/test/0"

# Función para manejar eventos de teclado
def on_press(key):
    global exit_program, capture_hand, output_dir   
    try:
        if key == keyboard.Key.ctrl:  # Salir del programa
            exit_program = True
        elif key == keyboard.Key.alt:  # Capturar mano
            capture_hand = True
        else:
            print(f"Tecla presionada: {key.char}")
            output_dir = f"dataset/test/{key.char.upper()}"
            print(f"Directorio de salida actualizado: {output_dir}")

    except AttributeError:
        pass

def on_release(key):
    global capture_hand
    if key == keyboard.Key.alt:
        capture_hand = False


expansion_factor = 0.06  # Valor inicial
# Función para manejar la rueda del ratón
# def on_scroll(x, y, dx, dy):
#     global expansion_factor
#     factor_change = 0.02
#     if dy > 0:
#         expansion_factor = min(0.5, expansion_factor + factor_change)  # Aumenta el factor
#     else:
#         expansion_factor = max(0.01, expansion_factor - factor_change)  # Reduce el factor
#     print(f"Nuevo expansion_factor: {expansion_factor:.2f}")
# mouse_listener = mouse.Listener(on_scroll=on_scroll)
# mouse_listener.start()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv.VideoCapture(2)

if not (cap.isOpened()):
    print("Could not open video device")
    exit()

frame_idx = 0
capture_delay = 0.1  # Por ejemplo, 2 segundos de espera
last_capture_time = 0  # Inicialmente no hay capturas previas


cv.namedWindow("preview")
os.makedirs(output_dir, exist_ok=True)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()



with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while cv.getWindowProperty('preview', cv.WND_PROP_VISIBLE) == 1:
        ret, frame = cap.read()
        if ret is False:
            print("No se pudo acceder a la cámara.")
            break
        
    

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        
        if not results.multi_hand_landmarks:
            frame = cv.flip(frame, 1)
            cv.imshow('preview',frame)
            if exit_program  or (cv.waitKey(1) & 0xFF == 27):
                break
            continue

        for hand_landmarks in results.multi_hand_landmarks:
        #    Dibujar los puntos de referencia y las conexiones
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calcular el bounding box de la mano
            h, w, _ = frame.shape
            # Calcular el bounding box expandido
            x_min = max(0, int((min([lm.x for lm in hand_landmarks.landmark]) - expansion_factor) * w))
            y_min = max(0, int((min([lm.y for lm in hand_landmarks.landmark]) - expansion_factor) * h))
            x_max = min(w, int((max([lm.x for lm in hand_landmarks.landmark]) + expansion_factor) * w))
            y_max = min(h, int((max([lm.y for lm in hand_landmarks.landmark]) + expansion_factor) * h))

            # Dibujar el bounding box
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            if capture_hand:
                current_time = time.time()  # Obtener el tiempo actual

                # Solo capturar si ha pasado el tiempo de espera
                if (current_time - last_capture_time) >= capture_delay:
                    hand_crop = frame[y_min:y_max, x_min:x_max]
                    if hand_crop.size > 0:
                        crop_path = os.path.join(output_dir, f"hand_{x_min}_{y_min}.jpg")
                        cv.imwrite(crop_path, hand_crop)
                        os.makedirs(output_dir, exist_ok=True)
                        print(f"Recorte guardado en: {crop_path}")
                        last_capture_time = current_time
        # Waits for a user input to quit the application
        frame = cv.flip(frame, 1)
        cv.imshow("preview", frame)
        if exit_program  or (cv.waitKey(1) & 0xFF == 27):
            break

        


      

cap.release()
cv.destroyAllWindows()
listener.stop()