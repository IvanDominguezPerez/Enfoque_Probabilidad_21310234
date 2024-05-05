import numpy as np

# Función para calcular la cinemática inversa de un robot de 2-DOF
def inverse_kinematics(x, y, L1, L2):
    # Calcular la distancia desde el origen al punto (x, y)
    d = np.sqrt(x**2 + y**2)

    # Verificar si el punto está dentro del espacio de trabajo del robot
    if d > L1 + L2 or d < np.abs(L1 - L2):
        print("El punto especificado está fuera del espacio de trabajo del robot.")
        return None

    # Calcular el ángulo de la primera articulación (q1)
    alpha = np.arctan2(y, x)
    beta = np.arccos((L1**2 + d**2 - L2**2) / (2 * L1 * d))
    q1 = alpha - beta

    # Calcular el ángulo de la segunda articulación (q2)
    gamma = np.arccos((L1**2 + L2**2 - d**2) / (2 * L1 * L2))
    q2 = np.pi - gamma

    return np.array([q1, q2])

# Parámetros del robot (longitudes de los eslabones)
L1 = 3
L2 = 2

# Coordenadas del punto deseado (posición del efector final)
x_desired = 2
y_desired = 3

# Calcular la cinemática inversa para el punto deseado
joint_angles = inverse_kinematics(x_desired, y_desired, L1, L2)

# Imprimir los ángulos de las articulaciones resultantes
if joint_angles is not None:
    print("Ángulo de la articulación 1 (q1):", np.degrees(joint_angles[0]))
    print("Ángulo de la articulación 2 (q2):", np.degrees(joint_angles[1]))
