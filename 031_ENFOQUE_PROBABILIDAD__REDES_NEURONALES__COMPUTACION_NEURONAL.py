#Practica: 031_ENFOQUE_PROBABILIDAD_REDES_NEURONALES_COMPUTACION_NEURONAL
#Alumno: IVAN_DOMINGUEZ
#Registro: 21310234
#Grupo: 7F1

# Importamos las bibliotecas necesarias
import numpy as np  # Para operaciones matemáticas y manejo de matrices
from tensorflow.keras.models import Sequential  # Para definir el modelo de red neuronal secuencial
from tensorflow.keras.layers import Dense, Input  # Para añadir capas y entradas a la red neuronal
from sklearn.datasets import load_iris  # Conjunto de datos de Iris
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y prueba
from sklearn.preprocessing import OneHotEncoder  # Para codificar las etiquetas de salida

# Cargamos el conjunto de datos Iris
data = load_iris()

# 'data.data' contiene las 4 características: largo y ancho del pétalo y sépalo.
X = data.data  # Características de entrada
# 'data.target' contiene las etiquetas de clase, indicando el tipo de flor.
y = data.target.reshape(-1, 1)  # Reshape para tenerlo en forma de columna

# Codificación One-Hot de las etiquetas (para que sea adecuado para clasificación).
encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y).toarray()

# Dividimos los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba).
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Creamos un modelo de red neuronal secuencial.
model = Sequential()

# Añadimos la capa de entrada usando la clase Input con el tamaño de las entradas.
model.add(Input(shape=(4,)))  # 4 entradas (características de las flores)

# Añadimos la primera capa de la red con 8 neuronas y activación ReLU.
model.add(Dense(8, activation='relu'))  # Capa oculta

# Añadimos otra capa oculta con 8 neuronas y activación ReLU.
model.add(Dense(8, activation='relu'))  # Otra capa oculta

# Añadimos la capa de salida con 3 neuronas (una por cada clase) y activación softmax.
model.add(Dense(3, activation='softmax'))  # Capa de salida para clasificación

# Compilamos el modelo.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamos el modelo con los datos de entrenamiento.
model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=1)

# Evaluamos el modelo con los datos de prueba, para verificar su precisión.
loss, accuracy = model.evaluate(X_test, y_test)

# Imprimimos el resultado de la precisión en el conjunto de prueba.
print(f'Precisión del modelo en el conjunto de prueba: {accuracy * 100:.2f}%')

#Este programa implementa una red neuronal simple para clasificar flores basándose en medidas de
#pétalos y sépalos. A través del entrenamiento, la red aprende a hacer predicciones sobre qué tipo de
#flor es, dado un conjunto de características. Este tipo de red neuronal es un ejemplo clásico de
#aprendizaje supervisado en el que se asignan etiquetas a entradas específicas.
