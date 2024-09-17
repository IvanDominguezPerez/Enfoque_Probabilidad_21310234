#Practica: 004_ENFOQUE_PROBABILIDAD_DISTRIBUCION_DE_PROBABILIDAD
#Alumno: IVAN_DOMINGUEZ
#Registro: 21310234
#Grupo: 7F1

import numpy as np  # Librería para trabajar con matrices y operaciones matemáticas
import matplotlib.pyplot as plt  # Librería para visualización de gráficos
from scipy.stats import norm  # Para trabajar con distribuciones normales

# Definimos la media y la desviación estándar de la altura en cm.
# En un grupo de personas, la altura generalmente sigue una distribución normal.
media = 170  # Altura promedio en cm
desviacion_estandar = 10  # Desviación estándar de la altura

# Simulamos la altura de 1000 personas usando una distribución normal.
# np.random.normal genera números aleatorios siguiendo una distribución normal.
alturas = np.random.normal(media, desviacion_estandar, 1000)

# Visualizamos los datos generados en un histograma para ver cómo se distribuyen.
plt.hist(alturas, bins=30, edgecolor='black', density=True, alpha=0.6)

# Para fines visuales, también podemos dibujar la curva de la distribución normal teórica
xmin, xmax = plt.xlim()  # Limites del gráfico en el eje X (altura)
x = np.linspace(xmin, xmax, 100)  # Genera 100 puntos entre xmin y xmax para la curva
p = norm.pdf(x, media, desviacion_estandar)  # Calcula la función de densidad de probabilidad de la normal
plt.plot(x, p, 'k', linewidth=2)  # Dibuja la curva de distribución normal teórica

# Añadimos títulos y etiquetas
plt.title('Distribución de alturas de personas')
plt.xlabel('Altura (cm)')
plt.ylabel('Probabilidad')

# Mostramos el gráfico
plt.show()

# Ahora calculamos la probabilidad de que una persona tenga una altura entre 160 y 180 cm.
# Utilizamos la función cdf (cumulative distribution function) que nos da la probabilidad acumulada.
probabilidad = norm.cdf(180, media, desviacion_estandar) - norm.cdf(160, media, desviacion_estandar)

# Imprimimos la probabilidad
print(f'La probabilidad de que una persona mida entre 160 cm y 180 cm es: {probabilidad * 100:.2f}%')

# También podemos calcular la probabilidad de una altura específica, como que alguien mida más de 190 cm.
# Para ello usamos la función complementaria de cdf.
probabilidad_mayor_190 = 1 - norm.cdf(190, media, desviacion_estandar)

# Imprimimos el resultado
print(f'La probabilidad de que una persona mida más de 190 cm es: {probabilidad_mayor_190 * 100:.2f}%')
