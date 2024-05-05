import cv2

# Cargar el clasificador pre-entrenado de Haar para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar la imagen de ejemplo
image = cv2.imread('imagen.jpg')

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectar rostros en la imagen utilizando el clasificador de Haar
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Dibujar rectángulos alrededor de los rostros detectados
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Mostrar la imagen con los rostros detectados
cv2.imshow('Rostros Detectados', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
