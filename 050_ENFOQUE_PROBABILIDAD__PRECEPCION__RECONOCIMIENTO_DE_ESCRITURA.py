import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Cargar el conjunto de datos MNIST (imágenes de dígitos escritos a mano)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocesamiento de datos
# Normalizar las imágenes y cambiar su forma para que sean adecuadas para la entrada de la red neuronal
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Codificar las etiquetas en un formato one-hot
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Construir el modelo de red neuronal convolucional (CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Evaluar el modelo con los datos de prueba
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Precisión en los datos de prueba:', test_acc)
