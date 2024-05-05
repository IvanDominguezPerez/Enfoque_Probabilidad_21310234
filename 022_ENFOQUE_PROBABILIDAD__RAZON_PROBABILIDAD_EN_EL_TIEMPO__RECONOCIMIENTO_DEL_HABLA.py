import numpy as np

class HMM:
    def __init__(self, states, observations, start_prob, transition_prob, emission_prob):
        """
        Inicializa un modelo oculto de Markov.

        Args:
        - states: Lista de estados posibles.
        - observations: Lista de observaciones posibles.
        - start_prob: Probabilidad inicial de los estados.
        - transition_prob: Matriz de probabilidades de transición entre estados.
        - emission_prob: Matriz de probabilidades de emisión de observaciones para cada estado.
        """
        self.states = states
        self.observations = observations
        self.start_prob = start_prob
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob

    def forward_algorithm(self, observations):
        """
        Aplica el algoritmo de avance (forward algorithm) para calcular la probabilidad de una secuencia de observaciones.

        Args:
        - observations: Secuencia de observaciones.

        Returns:
        - Probabilidad de la secuencia de observaciones dadas las probabilidades del modelo (forward probability).
        """
        num_states = len(self.states)
        num_observations = len(observations)
        forward_prob = np.zeros((num_observations, num_states))

        # Paso de inicialización
        for i in range(num_states):
            forward_prob[0, i] = self.start_prob[i] * self.emission_prob[i, self.observations.index(observations[0])]

        # Pasos de recursión
        for t in range(1, num_observations):
            for j in range(num_states):
                forward_prob[t, j] = np.sum(forward_prob[t - 1, i] * self.transition_prob[i, j] * self.emission_prob[j, self.observations.index(observations[t])] for i in range(num_states))

        # Paso de terminación
        forward_probability = np.sum(forward_prob[num_observations - 1, :])
        return forward_probability

# Definimos los estados y observaciones posibles
states = ['H', 'L']  # H: habla, L: silencio
observations = ['A', 'B']  # A: sonido, B: silencio

# Definimos las probabilidades iniciales
start_prob = np.array([0.5, 0.5])

# Definimos la matriz de probabilidades de transición
transition_prob = np.array([[0.7, 0.3],
                            [0.4, 0.6]])

# Definimos la matriz de probabilidades de emisión
emission_prob = np.array([[0.9, 0.1],
                           [0.2, 0.8]])

# Creamos una instancia del modelo HMM
hmm = HMM(states, observations, start_prob, transition_prob, emission_prob)

# Definimos una secuencia de observaciones
observations_sequence = ['A', 'B', 'A']

# Calculamos la probabilidad de la secuencia de observaciones utilizando el algoritmo de avance
forward_probability = hmm.forward_algorithm(observations_sequence)

# Imprimimos la probabilidad de la secuencia de observaciones
print("Probabilidad de la secuencia de observaciones:", forward_probability)
