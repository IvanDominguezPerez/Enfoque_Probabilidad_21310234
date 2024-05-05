import numpy as np
import matplotlib.pyplot as plt

# Definimos los posibles valores de la variable aleatoria
valores = np.array([1, 2, 3, 4, 5])

# Definimos las probabilidades asociadas a cada valor
probabilidades = np.array([0.1, 0.2, 0.3, 0.2, 0.2])

# Creamos la distribución de probabilidad
distribucion = dict(zip(valores, probabilidades))

# Visualizamos la distribución de probabilidad
plt.bar(distribucion.keys(), distribucion.values(), align='center')
plt.xlabel('Valores')
plt.ylabel('Probabilidades')
plt.title('Distribución de Probabilidad Discreta')
plt.show()

# Calculamos la esperanza (valor esperado) de la distribución
esperanza = np.sum(valores * probabilidades)
print("Esperanza:", esperanza)

# Calculamos la varianza de la distribución
varianza = np.sum((valores - esperanza) ** 2 * probabilidades)
print("Varianza:", varianza)
