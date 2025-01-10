import cv2
import mediapipe as mp
import math

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

while True:
    success, img = dispositivoCaptura.read()  # Captura la imagen de la cámara
    if not success:
        print("No se pudo acceder a la cámara.")
        break
    salir = False
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
                if id == 20:  # Punto 24 (ejemplo)
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    x20, y20 = cx, cy  # Guardar las coordenadas del punto 24
            mediaX = (x4 + x20) // 2 # Calcular la media de las coordenadas X
            mediaY = (y4 + y20) // 2 # Calcular la media de las coordenadas Y

            distanciaEntreDedos = math.hypot(x20 - x4, y20 - y4) # Calcular la distancia entre los dedos
            cv2.line(img, (x4, y4), (x20, y20), (0, 255, 0), 3) # Dibujar la linea entre los dedos

            if distanciaEntreDedos < 50:  # Si la distancia entre los dedos es menor a 50 pixeles
                salir = True
            if distanciaEntreDedos < 100:  # Si la distancia entre los dedos es menor a 100 pixeles
                #dibujar texto en la imagen
                cv2.putText(img, "Abriendo la puerta", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.imshow("Image", img)  # Mostrar la imagen en una ventana

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q') or salir:
        break

# Liberar recursos
dispositivoCaptura.release()
cv2.destroyAllWindows()
