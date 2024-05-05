import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise_covariance, measurement_noise_covariance):
        """
        Inicializa un filtro de Kalman.

        Args:
        - initial_state: El estado inicial del sistema.
        - initial_covariance: La covarianza inicial del estado.
        - process_noise_covariance: La covarianza del ruido del proceso.
        - measurement_noise_covariance: La covarianza del ruido de la medición.
        """
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise_covariance = process_noise_covariance
        self.measurement_noise_covariance = measurement_noise_covariance

    def predict(self, control_input=None):
        """
        Predice el próximo estado del sistema.

        Args:
        - control_input: La entrada de control opcional.

        Returns:
        - state_prediction: La predicción del próximo estado del sistema.
        """
        # Actualizamos el estado y la covarianza de acuerdo con el modelo de transición
        self.state = np.dot(self.transition_matrix, self.state)
        self.covariance = np.dot(np.dot(self.transition_matrix, self.covariance), self.transition_matrix.T) + self.process_noise_covariance

        return self.state

    def update(self, measurement):
        """
        Actualiza el estado y la covarianza del sistema basándose en una nueva medición.

        Args:
        - measurement: La nueva medición.

        Returns:
        - state_estimate: La estimación del estado actual del sistema.
        """
        # Calculamos la ganancia de Kalman
        kalman_gain = np.dot(np.dot(self.covariance, self.measurement_matrix.T), np.linalg.inv(np.dot(np.dot(self.measurement_matrix, self.covariance), self.measurement_matrix.T) + self.measurement_noise_covariance))
        
        # Actualizamos el estado y la covarianza basándonos en la nueva medición
        residual = measurement - np.dot(self.measurement_matrix, self.state)
        self.state = self.state + np.dot(kalman_gain, residual)
        self.covariance = np.dot((np.eye(self.state.shape[0]) - np.dot(kalman_gain, self.measurement_matrix)), self.covariance)

        return self.state

# Definimos los parámetros del filtro de Kalman
initial_state = np.array([[0], [0]])  # Estado inicial: posición y velocidad inicial
initial_covariance = np.eye(2)  # Covarianza inicial del estado
process_noise_covariance = 0.01 * np.eye(2)  # Covarianza del ruido del proceso
measurement_noise_covariance = 0.1  # Covarianza del ruido de la medición

# Definimos las matrices de transición y de medición (en este caso, un modelo de movimiento rectilíneo uniforme)
dt = 0.1  # Intervalo de tiempo entre las mediciones
transition_matrix = np.array([[1, dt], [0, 1]])  # Matriz de transición
measurement_matrix = np.array([[1, 0]])  # Matriz de medición

# Creamos una instancia del filtro de Kalman
kalman_filter = KalmanFilter(initial_state, initial_covariance, process_noise_covariance, measurement_noise_covariance)
kalman_filter.transition_matrix = transition_matrix
kalman_filter.measurement_matrix = measurement_matrix

# Simulamos el movimiento del objeto y realizamos mediciones
num_steps = 100
true_position = np.zeros(num_steps)
measurements = np.zeros(num_steps)
for t in range(num_steps):
    true_position[t] = 0.5 * t ** 2  # Movimiento cuadrático
    measurements[t] = true_position[t] + np.random.normal(0, np.sqrt(measurement_noise_covariance))

# Aplicamos el filtro de Kalman para estimar la posición del objeto
estimated_positions = np.zeros(num_steps)
for t in range(num_steps):
    kalman_filter.predict()
    kalman_filter.update(np.array([[measurements[t]]]))
    estimated_positions[t] = kalman_filter.state[0, 0]

# Imprimimos los resultados
print("Posición verdadera:", true_position)
print("Mediciones:", measurements)
print("Posición estimada por el filtro de Kalman:", estimated_positions)
