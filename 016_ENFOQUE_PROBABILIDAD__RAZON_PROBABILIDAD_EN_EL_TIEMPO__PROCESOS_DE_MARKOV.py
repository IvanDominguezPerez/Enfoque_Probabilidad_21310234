import numpy as np
import matplotlib.pyplot as plt

# Definimos la matriz de transición de Markov
transition_matrix = np.array([[0.8, 0.2],
                               [0.4, 0.6]])

# Definimos los estados posibles del proceso de Markov
states = [0, 1]

# Definimos el estado inicial
initial_state = 0

# Número de pasos de tiempo
num_steps = 50

# Creamos una lista para almacenar los estados en cada paso de tiempo
trajectory = [initial_state]

# Generamos la trayectoria del proceso de Markov
for _ in range(num_steps):
    current_state = trajectory[-1]
    next_state = np.random.choice(states, p=transition_matrix[current_state])
    trajectory.append(next_state)

# Graficamos la trayectoria del proceso de Markov
plt.plot(trajectory, marker='o')
plt.title('Trayectoria del Proceso de Markov')
plt.xlabel('Pasos de Tiempo')
plt.ylabel('Estado')
plt.xticks(np.arange(num_steps+1))
plt.grid(True)
plt.show()
