import numpy as np

class HiddenMarkovModel:
    def __init__(self, transition_matrix, emission_matrix, initial_distribution):
        """
        Inicializa un modelo oculto de Markov.

        Args:
        - transition_matrix: La matriz de transición de estado a estado.
        - emission_matrix: La matriz de emisión de estado a observación.
        - initial_distribution: La distribución inicial de los estados ocultos.
        """
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.initial_distribution = initial_distribution
        self.num_states = transition_matrix.shape[0]
        self.num_observations = emission_matrix.shape[1]

    def viterbi_algorithm(self, observations):
        """
        Implementación del algoritmo de Viterbi para decodificar la secuencia de estados ocultos a partir de una secuencia de observaciones.

        Args:
        - observations: Una lista de observaciones.

        Returns:
        - hidden_states_sequence: La secuencia de estados ocultos más probable dada la secuencia de observaciones.
        """
        num_observations = len(observations)
        # Inicializamos la matriz de probabilidades de Viterbi
        viterbi = np.zeros((num_observations, self.num_states))
        # Inicializamos la matriz de retroceso para almacenar las mejores rutas
        backpointer = np.zeros((num_observations, self.num_states), dtype=int)

        # Paso de inicialización
        viterbi[0] = self.initial_distribution * self.emission_matrix[:, observations[0]]

        # Paso de recursión
        for t in range(1, num_observations):
            for s in range(self.num_states):
                # Calculamos el valor de Viterbi y el estado anterior que maximiza la probabilidad
                viterbi[t, s] = np.max(viterbi[t - 1] * self.transition_matrix[:, s]) * self.emission_matrix[s, observations[t]]
                backpointer[t, s] = np.argmax(viterbi[t - 1] * self.transition_matrix[:, s])

        # Paso de terminación
        best_path_prob = np.max(viterbi[-1])
        best_last_state = np.argmax(viterbi[-1])

        # Reconstruimos la secuencia de estados ocultos
        hidden_states_sequence = [best_last_state]
        for t in range(num_observations - 2, -1, -1):
            hidden_states_sequence.insert(0, backpointer[t + 1, hidden_states_sequence[0]])

        return hidden_states_sequence, best_path_prob

# Definimos los parámetros del modelo oculto de Markov
transition_matrix = np.array([[0.7, 0.3],
                              [0.4, 0.6]])
emission_matrix = np.array([[0.2, 0.8],
                             [0.6, 0.4]])
initial_distribution = np.array([0.5, 0.5])
observations = [0, 1, 0]  # Observaciones observadas en cada paso de tiempo

# Creamos un modelo oculto de Markov
hmm = HiddenMarkovModel(transition_matrix, emission_matrix, initial_distribution)

# Aplicamos el algoritmo de Viterbi para decodificar la secuencia de estados ocultos
hidden_states_sequence, best_path_prob = hmm.viterbi_algorithm(observations)

# Imprimimos los resultados
print("Secuencia de estados ocultos más probable:", hidden_states_sequence)
print("Probabilidad de la mejor secuencia de estados ocultos:", best_path_prob)
