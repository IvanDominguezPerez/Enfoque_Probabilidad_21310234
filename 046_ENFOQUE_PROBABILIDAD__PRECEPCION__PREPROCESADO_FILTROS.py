import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('imagen.jpg')

# Mostrar la imagen original
cv2.imshow('Imagen Original', image)
cv2.waitKey(0)  # Esperar hasta que se presione una tecla
cv2.destroyAllWindows()  # Cerrar la ventana

# Aplicar un filtro de suavizado (filtro de media) a la imagen
kernel_size = 5  # Tamaño del kernel del filtro de media (debe ser un número impar)
smoothed_image = cv2.blur(image, (kernel_size, kernel_size))

# Mostrar la imagen suavizada
cv2.imshow('Imagen Suavizada', smoothed_image)
cv2.waitKey(0)  # Esperar hasta que se presione una tecla
cv2.destroyAllWindows()  # Cerrar la ventana
