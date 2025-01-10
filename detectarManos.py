import cv2
import mediapipe as mp
import math
import pyautogui  # Biblioteca para controlar el ratón

# Inicializar la cámara (0 para cámara integrada)
dispositivoCaptura = cv2.VideoCapture(0)

# Configuración de MediaPipe para detección de manos
mpManos = mp.solutions.hands  # Descargar el procesamiento de manos
manos = mpManos.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.9, 
    min_tracking_confidence=0.8
)

mpDibujar = mp.solutions.drawing_utils  # Para dibujar las conexiones de la mano
pantalla_ancho, pantalla_alto = pyautogui.size()  # Obtener las dimensiones de la pantalla

# Variable para rastrear el estado del clic
clic_sostenido = False

while True:
    success, img = dispositivoCaptura.read()  # Captura la imagen de la cámara
    if not success:
        print("No se pudo acceder a la cámara.")
        breakq

    img = cv2.flip(img, 1)  # Voltear la imagen horizontalmente (corregir efecto espejo)
    salir = False # Variable para salir del bucle
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir la imagen a RGB
    resultado = manos.process(imgRGB)  # Procesar la imagen con MediaPipe

    if resultado.multi_hand_landmarks:  # Si se detectan manos
        for handLms in resultado.multi_hand_landmarks:  # Para cada mano detectada
            mpDibujar.draw_landmarks(img, handLms, mpManos.HAND_CONNECTIONS)  # Dibujar las conexiones
            for id, lm in enumerate(handLms.landmark):  # Para cada punto de la mano
                alto, ancho, color = img.shape  # Obtener las dimensiones de la imagen
                cx, cy = int(lm.x * ancho), int(lm.y * alto)  # Coordenadas del punto
                
                # Dibujar círculos en puntos específicos
                if id == 4:  # Pulgar (ID 4)
                    cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
                    x4, y4 = cx, cy # Guardar las coordenadas del pulgar
                if id == 8:  # Punto 24 (ejemplo)
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    x8, y8 = cx, cy  # Guardar las coordenadas del punto 24
            # Punto medio entre pulgar e índice
            mediaX = (x4 + x8) // 2
            mediaY = (y4 + y8) // 2
            cv2.circle(img, (mediaX, mediaY), 10, (255, 255, 0), cv2.FILLED)

            # Mover el ratón según la posición del punto medio
            pantallaX = int(mediaX * pantalla_ancho / ancho)
            pantallaY = int(mediaY * pantalla_alto / alto)
            pyautogui.moveTo(pantallaX, pantallaY)

            # Calcular la distancia entre el pulgar y el índice
            distanciaEntreDedos = math.hypot(x8 - x4, y8 - y4)
            cv2.putText(img, f"Distancia: {int(distanciaEntreDedos)}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

               # Controlar el clic sostenido
            if distanciaEntreDedos < 30:  # Umbral para activar el clic sostenido
                if not clic_sostenido:  # Si el clic aún no está sostenido
                    pyautogui.mouseDown()
                    clic_sostenido = True
                    cv2.putText(img, "Clic Sostenido", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            else:  # Si los dedos se separan, liberar el clic
                if clic_sostenido:
                    pyautogui.mouseUp()
                    clic_sostenido = False
            
            #dibujar texto con el dedo pulgar
            # cv2.putText(img, f"Distancia entre dedos: {distanciaEntreDedos}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            #dibujar rastro de linea del meñique
            cv2.line(img, (x4, y4), (mediaX, mediaY), (0, 0, 255), 3)
            #dibujar rastro de linea del pulgar
            cv2.line(img, (mediaX, mediaY), (mediaX, mediaY), (0, 0, 255), 3)
    cv2.imshow("Image", img)  # Mostrar la imagen en una ventana

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q') or salir:
        break

# Liberar recursos
dispositivoCaptura.release()
cv2.destroyAllWindows()
