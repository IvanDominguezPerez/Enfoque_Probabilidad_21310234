import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicializa los pesos de las conexiones entre la capa de entrada y la capa oculta
        self.W1 = np.random.randn(input_size, hidden_size)
        # Inicializa los pesos de las conexiones entre la capa oculta y la capa de salida
        self.W2 = np.random.randn(hidden_size, output_size)
        # Inicializa los sesgos de la capa oculta
        self.b1 = np.zeros((1, hidden_size))
        # Inicializa los sesgos de la capa de salida
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        # Función de activación sigmoide
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # Propagación hacia adelante
        # Calcula la salida de la capa oculta
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        # Calcula la salida de la capa de salida
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def sigmoid_derivative(self, x):
        # Derivada de la función sigmoide
        return x * (1 - x)

    def backward(self, x, y, lr):
        # Propagación hacia atrás (backpropagation)
        # Calcula el error en la capa de salida
        self.error_output = y - self.a2
        # Calcula el gradiente de la capa de salida
        self.delta_output = self.error_output * self.sigmoid_derivative(self.a2)
        # Calcula el error en la capa oculta
        self.error_hidden = np.dot(self.delta_output, self.W2.T)
        # Calcula el gradiente de la capa oculta
        self.delta_hidden = self.error_hidden * self.sigmoid_derivative(self.a1)
        # Actualiza los pesos y sesgos
        self.W2 += np.dot(self.a1.T, self.delta_output) * lr
        self.b2 += np.sum(self.delta_output, axis=0) * lr
        self.W1 += np.dot(x.T, self.delta_hidden) * lr
        self.b1 += np.sum(self.delta_hidden, axis=0) * lr

    def train(self, X, y, epochs, lr):
        # Entrena la red neuronal
        for epoch in range(epochs):
            # Propagación hacia adelante y hacia atrás
            output = self.forward(X)
            self.backward(X, y, lr)
            # Calcula la pérdida en cada iteración
            loss = np.mean(np.square(y - output))
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de entrada y salida
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    # Crear y entrenar la red neuronal
    nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)
    nn.train(X, y, epochs=1000, lr=0.1)
