import numpy as np
import matplotlib.pyplot as plt

# Función para simular el movimiento del robot (modelo de transición de estado)
def robot_motion(x, u):
    # Modelo de movimiento simple: movimiento rectilíneo con velocidad constante
    dt = 1  # Intervalo de tiempo
    F = np.array([[1, dt],
                  [0, 1]])  # Matriz de transición de estado (modelo de movimiento)
    B = np.array([[0],
                  [1]])  # Matriz de control (en este caso, no hay control)
    return F @ x + B @ u

# Función para simular las mediciones del sensor (modelo de observación)
def sensor_measurement(x):
    # Modelo de observación simple: medición directa de la posición
    H = np.array([[1, 0]])  # Matriz de observación (modelo de observación)
    return H @ x

# Implementar el filtro de Kalman
def kalman_filter(measurements, initial_state, initial_covariance, process_noise_cov, measurement_noise_cov):
    # Inicializar el estado estimado y la covarianza estimada
    x_est = initial_state
    P_est = initial_covariance

    # Lista para almacenar las estimaciones de posición
    estimated_positions = []

    # Bucle principal del filtro de Kalman
    for z in measurements:
        # Predicción del estado
        x_pred = robot_motion(x_est, np.array([[0]]))
        P_pred = x_pred @ x_pred.T + process_noise_cov

        # Actualización del estado basado en la medición
        y = z - sensor_measurement(x_pred)  # Residual
        S = H @ P_pred @ H.T + measurement_noise_cov  # Covarianza de la medición
        K = P_pred @ H.T @ np.linalg.inv(S)  # Ganancia de Kalman
        x_est = x_pred + K @ y  # Actualización del estado estimado
        P_est = (np.eye(2) - K @ H) @ P_pred  # Actualización de la covarianza estimada

        # Almacenar la estimación de posición
        estimated_positions.append(x_est[0])

    return np.array(estimated_positions)

# Parámetros del filtro de Kalman
initial_state = np.array([[0], [0]])  # Estado inicial (posición y velocidad)
initial_covariance = np.eye(2) * 10  # Covarianza inicial
process_noise_cov = np.eye(2) * 0.1   # Covarianza del ruido del proceso (modelo de movimiento)
measurement_noise_cov = np.array([[0.5]])  # Covarianza del ruido de medición (modelo de observación)

# Generar mediciones simuladas
true_positions = np.linspace(0, 10, 50).reshape(-1, 1)  # Posiciones verdaderas del robot
measurements = true_positions + np.random.normal(0, 0.5, (50, 1))  # Mediciones del sensor con ruido

# Ejecutar el filtro de Kalman
estimated_positions = kalman_filter(measurements, initial_state, initial_covariance, process_noise_cov, measurement_noise_cov)

# Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.plot(true_positions, label='Posición Verdadera', color='blue')
plt.plot(measurements, 'rx', label='Mediciones', markersize=8)
plt.plot(estimated_positions, label='Estimación de Posición', color='green')
plt.xlabel('Tiempo')
plt.ylabel('Posición')
plt.title('Filtro de Kalman para Estimación de Posición')
plt.legend()
plt.grid(True)
plt.show()
