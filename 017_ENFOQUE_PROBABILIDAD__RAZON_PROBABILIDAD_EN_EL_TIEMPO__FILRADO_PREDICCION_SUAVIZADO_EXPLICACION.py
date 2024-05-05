import numpy as np

def forward_algorithm(observations, transition_matrix, emission_matrix, initial_distribution):
    """
    Implementación del algoritmo de retroceso hacia adelante para el filtrado en un modelo oculto de Markov.

    Args:
    - observations: Una lista de observaciones.
    - transition_matrix: La matriz de transición de estado a estado.
    - emission_matrix: La matriz de emisión de estado a observación.
    - initial_distribution: La distribución inicial de los estados ocultos.

    Returns:
    - filtered_beliefs: La distribución de creencias filtrada para los estados ocultos en cada paso de tiempo.
    """
    num_states = len(initial_distribution)
    num_observations = len(observations)
    filtered_beliefs = np.zeros((num_observations, num_states))

    # Paso de inicialización
    filtered_beliefs[0] = initial_distribution * emission_matrix[:, observations[0]]

    # Paso de predicción y corrección
    for t in range(1, num_observations):
        predicted_beliefs = np.dot(filtered_beliefs[t - 1], transition_matrix)
        filtered_beliefs[t] = predicted_beliefs * emission_matrix[:, observations[t]]
        filtered_beliefs[t] /= np.sum(filtered_beliefs[t])

    return filtered_beliefs

# Definimos los parámetros del modelo oculto de Markov
transition_matrix = np.array([[0.7, 0.3],
                              [0.4, 0.6]])
emission_matrix = np.array([[0.2, 0.8],
                             [0.6, 0.4]])
initial_distribution = np.array([0.5, 0.5])
observations = [0, 1, 0]  # Observaciones observadas en cada paso de tiempo

# Aplicamos el algoritmo de retroceso hacia adelante para filtrar las creencias
filtered_beliefs = forward_algorithm(observations, transition_matrix, emission_matrix, initial_distribution)

# Imprimimos los resultados
print("Distribución de creencias filtrada para los estados ocultos:")
print(filtered_beliefs)
