import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

# Carga y preprocesamiento del conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define el modelo de red neuronal
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Capa de aplanamiento para convertir imágenes 2D en un vector 1D
    Dense(128, activation='relu'),  # Capa densamente conectada con activación ReLU
    Dense(10)                       # Capa de salida con 10 neuronas (una por cada clase) y activación lineal
])

# Compila el modelo especificando la función de pérdida y el optimizador
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Entrena el modelo utilizando el conjunto de datos de entrenamiento
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evalúa el modelo en el conjunto de datos de prueba
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", test_accuracy)
