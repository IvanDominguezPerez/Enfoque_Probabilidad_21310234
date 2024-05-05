import numpy as np
import matplotlib.pyplot as plt

# Definir la función de movimiento del robot
def robot_motion(x, u, dt):
    # Modelo de movimiento del robot (simple)
    x[0] += u[0] * np.cos(x[2]) * dt
    x[1] += u[0] * np.sin(x[2]) * dt
    x[2] += u[1] * dt
    return x

# Definir la función de observación del robot
def robot_observation(x):
    # Modelo de observación del robot (simple)
    return np.array([x[0], x[1]])

# Definir la función de Jacobiano para el modelo de movimiento
def motion_jacobian(x, u, dt):
    # Jacobiano del modelo de movimiento
    jacobian = np.eye(3)
    jacobian[0, 2] = -u[0] * np.sin(x[2]) * dt
    jacobian[1, 2] = u[0] * np.cos(x[2]) * dt
    return jacobian

# Definir la función de Jacobiano para el modelo de observación
def observation_jacobian(x):
    # Jacobiano del modelo de observación
    jacobian = np.eye(2, 3)
    return jacobian

# Implementar el algoritmo SLAM
def slam(num_steps, u, z, x_init, P_init):
    # Inicializar el estado estimado y la covarianza
    x_est = x_init
    P_est = P_init

    # Inicializar el conjunto de puntos de referencia estimados
    landmarks = []

    # Bucle principal
    for t in range(num_steps):
        # Predicción del estado
        x_pred = robot_motion(x_est, u[:, t], dt)
        F = motion_jacobian(x_est, u[:, t], dt)
        P_pred = F @ P_est @ F.T + Q

        # Actualización del estado
        H = observation_jacobian(x_pred)
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        x_est = x_pred + K @ (z[:, t] - robot_observation(x_pred))
        P_est = (np.eye(3) - K @ H) @ P_pred

        # Almacenar la posición estimada del robot
        estimated_positions.append(x_est[:2])

    return np.array(landmarks), np.array(estimated_positions)

# Parámetros del entorno
num_steps = 100
dt = 0.1

# Parámetros del filtro de Kalman extendido (EKF)
Q = np.diag([0.1, 0.1, 0.01])
R = np.diag([0.1, 0.1])

# Estado inicial del robot [x, y, theta]
x_init = np.array([0, 0, np.pi/4])

# Covarianza inicial del estado
P_init = np.diag([1, 1, np.pi/4])

# Secuencia de comandos de control (velocidad lineal y velocidad angular)
u = np.vstack((np.ones(num_steps)*2, np.ones(num_steps)*0.1))

# Lecturas del sensor (observaciones)
z = np.random.normal(loc=0, scale=0.5, size=(2, num_steps))

# Ejecutar el algoritmo SLAM
landmarks, estimated_positions = slam(num_steps, u, z, x_init, P_init)

# Visualizar el mapa generado por SLAM
plt.figure(figsize=(10, 6))
plt.plot(landmarks[:, 0], landmarks[:, 1], 'ro', label='Landmarks Reales')
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'b-', label='Posición Estimada del Robot')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mapa Generado por SLAM')
plt.legend()
plt.grid(True)
plt.show()
