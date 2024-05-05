import numpy as np
import matplotlib.pyplot as plt

# Creamos una distribución de probabilidad discreta para representar la incertidumbre
valores = np.array([0, 1, 2, 3, 4])  # Valores posibles
probabilidades = np.array([0.1, 0.2, 0.4, 0.2, 0.1])  # Probabilidades correspondientes

# Visualizamos la distribución de probabilidad
plt.bar(valores, probabilidades, align='center')
plt.xlabel('Valor')
plt.ylabel('Probabilidad')
plt.title('Distribución de Probabilidad')
plt.show()

# Calculamos la esperanza (valor esperado) de la distribución
esperanza = np.sum(valores * probabilidades)
print("Esperanza:", esperanza)

# Calculamos la varianza de la distribución
varianza = np.sum((valores - esperanza) ** 2 * probabilidades)
print("Varianza:", varianza)
