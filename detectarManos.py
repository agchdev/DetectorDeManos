import cv2
import mediapipe as mp

dispositivoCaptura = cv2.VideoCapture(2)

mpManos=mp.solutions.hands # Descarga el procesamiento de las manos
manos=mpManos.Hands(static_image_mode = False, max_num_hands = 1, min_detection_confidence = 0.9, min_tracking_confidence = 0.8)
# Los valores van del 0 al 1 y quieren decir lo siente
# static_image_mode = False -> Esto quiere decir que se va a detectar la mano en tiempo real
# max_num_hands = 1 -> Esto quiere decir que se va a detectar una sola mano
# min_detection_confidence = 0.9 -> Esto quiere decir que la probabilidad de que se detecte la mano es del 90%
# min_tracking_confidence = 0.8 -> Esto quiere decir que la probabilidad de que se detecte la mano es del 80%

mpDibujar=mp.solutions.drawing_utils

# mpDibujar=mp.solutions.drawing_utils -> Esto quiere decir que se va a dibujar las manos

while True:
    success,img = dispositivoCaptura.read() # Captura la imagen
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convierte la imagen a RGB
    resultado = manos.process(imgRGB) # Procesa la imagen
    # print(resultado.multi_hand_landmarks) # Imprime los puntos de la mano

    if resultado.multi_hand_landmarks: # Si se detecta una mano
        for handLms in resultado.multi_hand_landmarks: # Para cada mano
            # mpDibujar.draw_landmarks(img, handLms, mpManos.HAND_CONNECTIONS) # Dibuja la mano
            for id, lm in enumerate(handLms.landmark): # Para cada punto de la mano
                alto,ancho,color = img.shape # Obtiene el alto, ancho y color de la imagen
                cx,cy = int(lm.x * ancho), int(lm.y * alto) # Obtiene el centro de la mano
                if id == 4: # Si el punto es el 4
                    cv2.circle(img,(cx,cy),10,(255,255,0),cv2.FILLED) # Dibuja un circulo azul
                    #cv2.circle -> Dibuja un circulo en la imagen
                    #img -> La imagen
                    #(cx,cy) -> El centro del circulo
                    #10 -> El radio del circulo
                    #(255,255,0) -> El color del circulo
                    #cv2.FILLED -> El circulo se llena
                if id == 24: # Si el punto es el 24
                    cv2.circle(img,(cx,cy),10,(255,255,0),cv2.FILLED) # Dibuja un circulo azul


    cv2.imshow("Image",img) # Muestra la imagen
    cv2.waitKey(1) # Espera 1 milisegundo