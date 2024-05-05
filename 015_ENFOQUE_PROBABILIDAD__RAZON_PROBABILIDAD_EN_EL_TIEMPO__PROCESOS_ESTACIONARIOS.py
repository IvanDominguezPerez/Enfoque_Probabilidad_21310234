import numpy as np
import matplotlib.pyplot as plt

# Definimos los parámetros del proceso estacionario
mu = 0         # Media
sigma = 1      # Desviación estándar
num_samples = 1000  # Número de muestras
num_steps = 100     # Número de pasos de tiempo

# Generamos muestras del proceso estacionario
samples = np.random.normal(mu, sigma, size=(num_samples, num_steps))

# Calculamos la media y la autocorrelación a lo largo del tiempo
mean = np.mean(samples, axis=0)
autocorr = np.correlate(samples[0], samples[0], mode='full')

# Graficamos las muestras, la media y la autocorrelación
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
for i in range(num_samples):
    plt.plot(samples[i], color='gray', alpha=0.5)
plt.plot(mean, color='blue', linewidth=2, label='Media')
plt.title('Muestras del Proceso Estacionario')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(autocorr)
plt.title('Autocorrelación del Proceso Estacionario')
plt.xlabel('Desplazamiento')
plt.ylabel('Autocorrelación')

plt.tight_layout()
plt.show()
