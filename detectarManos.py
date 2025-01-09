import cv2
import mediapipe as mp

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

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir la imagen a RGB
    resultado = manos.process(imgRGB)  # Procesar la imagen con MediaPipe

    if resultado.multi_hand_landmarks:  # Si se detectan manos
        for handLms in resultado.multi_hand_landmarks:  # Para cada mano detectada
            mpDibujar.draw_landmarks(img, handLms, mpManos.HAND_CONNECTIONS)  # Dibujar las conexiones

            for id, lm in enumerate(handLms.landmark):  # Para cada punto de la mano
                alto, ancho, _ = img.shape  # Obtener las dimensiones de la imagen
                cx, cy = int(lm.x * ancho), int(lm.y * alto)  # Coordenadas del punto
                
                # Dibujar círculos en puntos específicos
                if id == 4:  # Pulgar (ID 4)
                    cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
                if id == 24:  # Punto 24 (ejemplo)
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Image", img)  # Mostrar la imagen en una ventana

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
dispositivoCaptura.release()
cv2.destroyAllWindows()
