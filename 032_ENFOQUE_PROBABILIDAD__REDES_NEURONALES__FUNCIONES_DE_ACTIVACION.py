import numpy as np

class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        """
        Función de activación sigmoide.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """
        Derivada de la función sigmoide.
        """
        return x * (1 - x)

    @staticmethod
    def relu(x):
        """
        Función de activación ReLU (Rectified Linear Unit).
        """
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """
        Derivada de la función ReLU.
        """
        return np.where(x <= 0, 0, 1)

    @staticmethod
    def tanh(x):
        """
        Función de activación tangente hiperbólica.
        """
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        """
        Derivada de la función tangente hiperbólica.
        """
        return 1 - np.tanh(x)**2

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo
    x = np.array([-2, -1, 0, 1, 2])

    # Función de activación sigmoide
    sigmoid_output = ActivationFunctions.sigmoid(x)
    sigmoid_derivative_output = ActivationFunctions.sigmoid_derivative(sigmoid_output)
    print("Sigmoid Output:", sigmoid_output)
    print("Sigmoid Derivative Output:", sigmoid_derivative_output)

    # Función de activación ReLU
    relu_output = ActivationFunctions.relu(x)
    relu_derivative_output = ActivationFunctions.relu_derivative(relu_output)
    print("ReLU Output:", relu_output)
    print("ReLU Derivative Output:", relu_derivative_output)

    # Función de activación tangente hiperbólica
    tanh_output = ActivationFunctions.tanh(x)
    tanh_derivative_output = ActivationFunctions.tanh_derivative(tanh_output)
    print("Tanh Output:", tanh_output)
    print("Tanh Derivative Output:", tanh_derivative_output)
