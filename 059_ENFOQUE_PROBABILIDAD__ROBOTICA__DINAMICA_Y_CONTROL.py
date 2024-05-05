import numpy as np
import matplotlib.pyplot as plt

# Función para simular el movimiento del robot (modelo de dinámica)
def robot_dynamics(x, u):
    # Modelo de dinámica simple: movimiento rectilíneo con velocidad constante
    dt = 0.1  # Intervalo de tiempo
    return x + u * dt

# Función para el controlador proporcional (P)
def proportional_controller(x_desired, x_current, Kp):
    # Error entre la posición deseada y la posición actual
    error = x_desired - x_current
    # Ley de control proporcional
    u = Kp * error
    return u

# Parámetros del robot
x_initial = 0  # Posición inicial del robot
x_desired = 5  # Posición deseada del robot
Kp = 1         # Ganancia del controlador proporcional

# Listas para almacenar los estados y las acciones del robot
x_states = [x_initial]
u_actions = []

# Simulación del control del robot
for _ in range(50):  # Simular durante 50 pasos de tiempo
    # Calcular la acción de control utilizando el controlador proporcional
    u = proportional_controller(x_desired, x_states[-1], Kp)
    # Aplicar la acción de control y actualizar el estado del robot
    x_states.append(robot_dynamics(x_states[-1], u))
    u_actions.append(u)

# Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.plot(x_states, label='Posición del Robot')
plt.plot([0, len(x_states)], [x_desired, x_desired], 'r--', label='Posición Deseada')
plt.xlabel('Tiempo')
plt.ylabel('Posición')
plt.title('Control de Robot utilizando Controlador Proporcional')
plt.legend()
plt.grid(True)
plt.show()
