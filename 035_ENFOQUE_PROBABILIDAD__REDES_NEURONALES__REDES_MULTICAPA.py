import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Cargar el conjunto de datos MNIST y dividirlo en datos de entrenamiento y prueba
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los valores de píxel al rango [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convertir las etiquetas a codificación one-hot
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Definir la arquitectura de la red neuronal
model = Sequential([
    Flatten(input_shape=(28, 28)),     # Capa de entrada: aplanar la imagen de 28x28 píxeles
    Dense(128, activation='relu'),     # Capa oculta con 128 neuronas y función de activación ReLU
    Dense(10, activation='softmax')    # Capa de salida con 10 neuronas (una por clase) y función de activación softmax
])

# Compilar el modelo
model.compile(optimizer='adam',       # Algoritmo de optimización: Adam
              loss='categorical_crossentropy',   # Función de pérdida: entropía cruzada categórica
              metrics=['accuracy'])              # Métrica a monitorear: precisión

# Entrenar el modelo
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Precisión en el conjunto de prueba:", test_acc)
