import numpy as np
import matplotlib.pyplot as plt

# Creamos datos de ejemplo para dos clases linealmente separables
np.random.seed(0)
num_samples = 100
class_1 = np.random.randn(num_samples, 2) + np.array([2, 2])
class_2 = np.random.randn(num_samples, 2) + np.array([-2, -2])

# Concatenamos los datos y etiquetamos las clases
X = np.vstack([class_1, class_2])
y = np.hstack([np.ones(num_samples), np.zeros(num_samples)])

# Visualizamos los datos
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
plt.title('Datos de ejemplo')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.show()

# Definimos la función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Inicializamos los pesos y el sesgo
np.random.seed(0)
weights = np.random.randn(2)
bias = np.random.randn()

# Definimos la función de predicción
def predict(X):
    z = np.dot(X, weights) + bias
    return sigmoid(z)

# Entrenamos el modelo usando el descenso de gradiente
learning_rate = 0.1
num_epochs = 1000

for epoch in range(num_epochs):
    # Calculamos las predicciones
    predictions = predict(X)
    
    # Calculamos el error
    error = y - predictions
    
    # Actualizamos los pesos y el sesgo utilizando el descenso de gradiente
    weights += learning_rate * np.dot(X.T, error)
    bias += learning_rate * np.sum(error)

# Visualizamos la línea de separación obtenida
x_values = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
y_values = -(weights[0] * x_values + bias) / weights[1]

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
plt.plot(x_values, y_values, color='k', linestyle='-', linewidth=2)
plt.title('Separabilidad lineal obtenida')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.show()
