import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar detección de bordes utilizando el operador de Sobel
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Sobel en dirección x
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Sobel en dirección y
edges = np.sqrt(sobel_x**2 + sobel_y**2)  # Magnitud del gradiente

# Aplicar umbralización para segmentar los bordes
threshold_value = 100  # Valor de umbral
edges_binary = np.uint8(edges > threshold_value) * 255  # Umbralización binaria

# Mostrar la imagen original
cv2.imshow('Imagen Original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Mostrar los bordes detectados
cv2.imshow('Bordes Detectados', edges_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
