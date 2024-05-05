import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, x):
        summation = np.dot(x, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(y.shape[0]):
                prediction = self.predict(X[i])
                self.weights[1:] += self.learning_rate * (y[i] - prediction) * X[i]
                self.weights[0] += self.learning_rate * (y[i] - prediction)

class Adaline:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, x):
        summation = np.dot(x, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(y.shape[0]):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights[1:] += self.learning_rate * error * X[i]
                self.weights[0] += self.learning_rate * error

class Madaline:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, x):
        summation = np.dot(x, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(y.shape[0]):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                if error != 0:
                    self.weights[1:] += self.learning_rate * error * X[i]
                    self.weights[0] += self.learning_rate * error

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de entrenamiento
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y_AND = np.array([0, 0, 0, 1])
    y_OR = np.array([0, 1, 1, 1])
    y_XOR = np.array([0, 1, 1, 0])

    # Crear y entrenar los modelos
    print("Perceptron (AND):")
    perceptron_AND = Perceptron(input_size=2)
    perceptron_AND.train(X, y_AND)
    print("Weights:", perceptron_AND.weights)

    print("\nPerceptron (OR):")
    perceptron_OR = Perceptron(input_size=2)
    perceptron_OR.train(X, y_OR)
    print("Weights:", perceptron_OR.weights)

    print("\nAdaline (AND):")
    adaline_AND = Adaline(input_size=2)
    adaline_AND.train(X, y_AND)
    print("Weights:", adaline_AND.weights)

    print("\nMadaline (XOR):")
    madaline_XOR = Madaline(input_size=2)
    madaline_XOR.train(X, y_XOR)
    print("Weights:", madaline_XOR.weights)
