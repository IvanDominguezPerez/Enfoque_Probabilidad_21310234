import numpy as np
import matplotlib.pyplot as plt

class ParticleFilter:
    def __init__(self, num_particles, transition_function, measurement_function, initial_particles, process_noise_covariance, measurement_noise_covariance):
        """
        Inicializa un filtro de partículas.

        Args:
        - num_particles: El número de partículas.
        - transition_function: La función de transición que predice el próximo estado.
        - measurement_function: La función de medición que mapea el estado a la medición.
        - initial_particles: Las partículas iniciales.
        - process_noise_covariance: La covarianza del ruido del proceso.
        - measurement_noise_covariance: La covarianza del ruido de la medición.
        """
        self.num_particles = num_particles
        self.transition_function = transition_function
        self.measurement_function = measurement_function
        self.particles = initial_particles
        self.process_noise_covariance = process_noise_covariance
        self.measurement_noise_covariance = measurement_noise_covariance

    def predict(self):
        """
        Predice el próximo estado de todas las partículas según la función de transición y agrega ruido del proceso.
        """
        for i in range(self.num_particles):
            # Predice el próximo estado utilizando la función de transición y agrega ruido del proceso
            self.particles[i] = self.transition_function(self.particles[i]) + np.random.multivariate_normal(mean=np.zeros(len(self.particles[i])), cov=self.process_noise_covariance)

    def update(self, measurement):
        """
        Actualiza las ponderaciones de las partículas basadas en la nueva medición.

        Args:
        - measurement: La nueva medición.
        """
        # Calcula las ponderaciones de las partículas basadas en la función de medición
        weights = np.array([self.measurement_function(particle, measurement, self.measurement_noise_covariance) for particle in self.particles])

        # Normaliza las ponderaciones
        weights /= np.sum(weights)

        # Resamplea las partículas según las ponderaciones
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=weights)
        self.particles = self.particles[indices]

# Definimos la función de transición
def transition_function(x):
    return np.array([x[0] + 1, x[1] + np.random.normal(0, 0.1)])

# Definimos la función de medición
def measurement_function(x, measurement, measurement_noise_covariance):
    return np.exp(-np.sum((x - measurement) ** 2) / (2 * measurement_noise_covariance))

# Definimos los parámetros del filtro de partículas
num_particles = 100
initial_particles = np.random.normal(loc=np.array([0, 0]), scale=np.array([1, 1]), size=(num_particles, 2))
process_noise_covariance = np.diag([0.1, 0.1])
measurement_noise_covariance = 0.1

# Creamos una instancia del filtro de partículas
particle_filter = ParticleFilter(num_particles, transition_function, measurement_function, initial_particles, process_noise_covariance, measurement_noise_covariance)

# Simulamos el movimiento del objeto y las mediciones
true_positions = np.zeros((100, 2))
measurements = np.zeros((100, 2))
for t in range(100):
    true_positions[t] = [t, np.sin(t)]
    measurements[t] = true_positions[t] + np.random.normal(0, np.sqrt(measurement_noise_covariance), size=2)

# Aplicamos el filtro de partículas para estimar la posición del objeto
estimated_positions = np.zeros((100, 2))
for t in range(100):
    particle_filter.predict()
    particle_filter.update(measurements[t])
    estimated_positions[t] = np.mean(particle_filter.particles, axis=0)

# Graficamos los resultados
plt.figure(figsize=(10, 6))
plt.plot(true_positions[:, 0], true_positions[:, 1], label="Posición Verdadera")
plt.scatter(measurements[:, 0], measurements[:, 1], color='r', label="Mediciones")
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], color='g', label="Posición Estimada (Filtro de Partículas)")
plt.xlabel("Tiempo")
plt.ylabel("Posición")
plt.title("Filtrado de Partículas para Red Bayesiana Dinámica")
plt.legend()
plt.grid(True)
plt.show()
