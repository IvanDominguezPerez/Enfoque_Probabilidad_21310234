import numpy as np

# Definir la función de activación sigmoide y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Datos de entrada
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Etiquetas de salida esperadas
y = np.array([[0],
              [1],
              [1],
              [0]])

# Inicializar pesos y sesgos aleatoriamente
np.random.seed(1)
input_neurons = 2
hidden_neurons = 3
output_neurons = 1

# Pesos para la capa oculta
weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
# Pesos para la capa de salida
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

# Entrenamiento
learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    # Feedforward
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)
    
    # Cálculo del error
    error = y - output_layer_output
    
    # Retropropagación
    d_output = error * sigmoid_derivative(output_layer_output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Actualización de pesos
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

# Evaluación
hidden_layer_input = np.dot(X, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
output_layer_output = sigmoid(output_layer_input)

print("Salida después del entrenamiento:")
print(output_layer_output)
