import numpy as np

class HammingNetwork:
    def __init__(self, patterns):
        self.weights = np.zeros((len(patterns[0]), len(patterns[0])))
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)

    def recall(self, pattern, max_iterations=10):
        prev_pattern = pattern.copy()
        for _ in range(max_iterations):
            pattern = np.sign(np.dot(pattern, self.weights))
            if np.array_equal(pattern, prev_pattern):
                break
            prev_pattern = pattern.copy()
        return pattern

class HopfieldNetwork:
    def __init__(self, patterns):
        self.weights = np.zeros((len(patterns[0]), len(patterns[0])))
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, max_iterations=10):
        prev_pattern = pattern.copy()
        for _ in range(max_iterations):
            pattern = np.sign(np.dot(pattern, self.weights))
            if np.array_equal(pattern, prev_pattern):
                break
            prev_pattern = pattern.copy()
        return pattern

class HebbianNetwork:
    def __init__(self):
        pass

    def train(self, patterns):
        num_neurons = len(patterns[0])
        self.weights = np.zeros((num_neurons, num_neurons))
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)

    def recall(self, pattern):
        return np.sign(np.dot(pattern, self.weights))

class BoltzmannMachine:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns, learning_rate=0.1, num_epochs=100):
        for _ in range(num_epochs):
            for pattern in patterns:
                self.weights += learning_rate * np.outer(pattern, pattern) - np.eye(self.num_neurons)

    def recall(self, pattern, max_iterations=10):
        prev_pattern = pattern.copy()
        for _ in range(max_iterations):
            pattern = np.sign(np.dot(pattern, self.weights))
            if np.array_equal(pattern, prev_pattern):
                break
            prev_pattern = pattern.copy()
        return pattern

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo (patrones)
    patterns = np.array([
        [1, 1, 1, -1, -1],
        [-1, -1, 1, 1, 1],
        [1, -1, -1, -1, 1]
    ])

    # Hamming Network
    hamming_net = HammingNetwork(patterns)
    print("Hamming Network:")
    for pattern in patterns:
        print("Pattern:", pattern, "Recall:", hamming_net.recall(pattern))

    # Hopfield Network
    hopfield_net = HopfieldNetwork(patterns)
    print("\nHopfield Network:")
    for pattern in patterns:
        print("Pattern:", pattern, "Recall:", hopfield_net.recall(pattern))

    # Hebbian Network
    hebb_net = HebbianNetwork()
    hebb_net.train(patterns)
    print("\nHebbian Network:")
    for pattern in patterns:
        print("Pattern:", pattern, "Recall:", hebb_net.recall(pattern))

    # Boltzmann Machine
    boltzmann_machine = BoltzmannMachine(len(patterns[0]))
    boltzmann_machine.train(patterns)
    print("\nBoltzmann Machine:")
    for pattern in patterns:
        print("Pattern:", pattern, "Recall:", boltzmann_machine.recall(pattern))
