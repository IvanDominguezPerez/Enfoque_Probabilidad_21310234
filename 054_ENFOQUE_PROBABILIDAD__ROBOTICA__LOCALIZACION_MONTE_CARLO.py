import numpy as np
import matplotlib.pyplot as plt

# Función para generar partículas aleatorias
def generate_particles(num_particles, x_range, y_range):
    particles = []
    for _ in range(num_particles):
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        particles.append([x, y])
    return np.array(particles)

# Función para visualizar las partículas
def plot_particles(particles):
    plt.scatter(particles[:, 0], particles[:, 1], color='blue', marker='o')
    plt.title('Distribución de Partículas')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

# Parámetros del entorno y las mediciones
num_particles = 1000  # Número de partículas
x_range = (0, 10)     # Rango en el eje X
y_range = (0, 10)     # Rango en el eje Y
measurement = [5, 5]  # Mediciones de posición (en este caso, las coordenadas del robot)

# Generar partículas aleatorias dentro del rango especificado
particles = generate_particles(num_particles, x_range, y_range)

# Visualizar las partículas inicialmente
plot_particles(particles)

# Actualizar la distribución de partículas en base a las mediciones
# (en este ejemplo, simplemente se desplazan las partículas hacia la posición de la medición)
particles += np.random.normal(0, 1, size=particles.shape)  # Agregar ruido gaussiano
particles += (measurement - particles) * 0.1               # Desplazar las partículas hacia la medición

# Visualizar las partículas después de la actualización
plot_particles(particles)
