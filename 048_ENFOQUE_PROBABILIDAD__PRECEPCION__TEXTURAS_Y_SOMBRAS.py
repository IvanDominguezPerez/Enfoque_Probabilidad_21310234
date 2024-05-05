import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('imagen.jpg')

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calcular el histograma de gradientes orientados (HOG) para la detección de texturas
hog = cv2.HOGDescriptor()
features, hog_image = hog.compute(gray_image)

# Mostrar la imagen original y el resultado del HOG
cv2.imshow('Imagen Original', image)
cv2.imshow('Resultado del HOG', hog_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Aplicar una sombra simple a la imagen
shadow_image = cv2.addWeighted(image, 0.5, np.zeros_like(image), 0.5, 0)

# Mostrar la imagen original y la imagen con sombra
cv2.imshow('Imagen Original', image)
cv2.imshow('Imagen con Sombra', shadow_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
