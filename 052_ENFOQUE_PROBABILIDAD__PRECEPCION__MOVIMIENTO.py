import cv2
import numpy as np

# Función para calcular el flujo óptico utilizando Lucas-Kanade
def optical_flow_lucas_kanade(previous_frame, current_frame, points):
    # Calcular el flujo óptico utilizando el método de Lucas-Kanade
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_frame, current_frame, points, None)
    
    # Seleccionar solo los puntos con un buen seguimiento (status = 1)
    good_points = next_points[status == 1]
    
    return good_points

# Capturar video desde la cámara web (puedes cambiar el número de la cámara o utilizar un archivo de video)
cap = cv2.VideoCapture(0)

# Leer el primer frame para inicializar el seguimiento
ret, previous_frame = cap.read()
previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

# Detectar puntos característicos en el primer frame (puedes cambiar los parámetros según tus necesidades)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
points = cv2.goodFeaturesToTrack(previous_gray, mask=None, **feature_params)

# Bucle principal para el seguimiento de movimiento en tiempo real
while True:
    # Leer el siguiente frame del video
    ret, current_frame = cap.read()
    if not ret:
        break
    
    # Convertir el frame a escala de grises
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Calcular el flujo óptico entre los dos frames
    points = optical_flow_lucas_kanade(previous_gray, current_gray, points)
    
    # Dibujar los puntos en el frame actual
    for point in points:
        x, y = point.ravel()
        cv2.circle(current_frame, (x, y), 5, (0, 255, 0), -1)
    
    # Mostrar el frame con los puntos detectados
    cv2.imshow('Movimiento', current_frame)
    
    # Actualizar el frame anterior y los puntos para el siguiente bucle
    previous_gray = current_gray.copy()
    points = np.float32(points).reshape(-1, 1, 2)
    
    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
