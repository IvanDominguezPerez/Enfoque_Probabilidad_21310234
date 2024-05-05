import numpy as np

def forward_backward_algorithm(observations, num_states, transition_matrix, emission_matrix, initial_distribution, max_iterations=100, tol=1e-4):
    """
    Implementación del algoritmo hacia adelante-atrás (Baum-Welch) para estimar los parámetros de un modelo oculto de Markov.

    Args:
    - observations: Una lista de observaciones.
    - num_states: El número de estados en el modelo oculto de Markov.
    - transition_matrix: La matriz de transición de estado a estado.
    - emission_matrix: La matriz de emisión de estado a observación.
    - initial_distribution: La distribución inicial de los estados ocultos.
    - max_iterations: El número máximo de iteraciones del algoritmo.
    - tol: La tolerancia para la convergencia del algoritmo.

    Returns:
    - transition_matrix: La matriz de transición estimada.
    - emission_matrix: La matriz de emisión estimada.
    - initial_distribution: La distribución inicial estimada.
    """
    num_observations = len(observations)

    for _ in range(max_iterations):
        # Paso hacia adelante
        alpha = np.zeros((num_observations, num_states))
        alpha[0] = initial_distribution * emission_matrix[:, observations[0]]
        for t in range(1, num_observations):
            alpha[t] = np.dot(alpha[t - 1], transition_matrix) * emission_matrix[:, observations[t]]

        # Paso hacia atrás
        beta = np.ones((num_observations, num_states))
        for t in range(num_observations - 2, -1, -1):
            beta[t] = np.dot(transition_matrix, beta[t + 1] * emission_matrix[:, observations[t + 1]])

        # Estimación de los parámetros
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1)[:, np.newaxis]

        xi = np.zeros((num_observations - 1, num_states, num_states))
        for t in range(num_observations - 1):
            xi[t] = alpha[t][:, np.newaxis] * transition_matrix * emission_matrix[:, observations[t + 1]] * beta[t + 1]
        xi /= np.sum(xi, axis=(1, 2))[:, np.newaxis, np.newaxis]

        # Actualización de los parámetros
        new_initial_distribution = gamma[0]
        new_transition_matrix = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, np.newaxis]
        new_emission_matrix = np.copy(emission_matrix)
        for k in range(num_states):
            new_emission_matrix[k] = np.sum(gamma[observations == k], axis=0) / np.sum(gamma, axis=0)[k]

        # Verificación de la convergencia
        if np.max(np.abs(new_transition_matrix - transition_matrix)) < tol and \
           np.max(np.abs(new_emission_matrix - emission_matrix)) < tol and \
           np.max(np.abs(new_initial_distribution - initial_distribution)) < tol:
            break

        transition_matrix = new_transition_matrix
        emission_matrix = new_emission_matrix
        initial_distribution = new_initial_distribution

    return transition_matrix, emission_matrix, initial_distribution

# Definimos los parámetros del modelo oculto de Markov
num_states = 2
transition_matrix = np.array([[0.7, 0.3],
                              [0.4, 0.6]])
emission_matrix = np.array([[0.2, 0.8],
                             [0.6, 0.4]])
initial_distribution = np.array([0.5, 0.5])
observations = [0, 1, 0]  # Observaciones observadas en cada paso de tiempo

# Aplicamos el algoritmo hacia adelante-atrás para estimar los parámetros del modelo oculto de Markov
estimated_transition_matrix, estimated_emission_matrix, estimated_initial_distribution = forward_backward_algorithm(
    observations, num_states, transition_matrix, emission_matrix, initial_distribution)

# Imprimimos los resultados
print("Matriz de transición estimada:")
print(estimated_transition_matrix)
print("Matriz de emisión estimada:")
print(estimated_emission_matrix)
print("Distribución inicial estimada:")
print(estimated_initial_distribution)
