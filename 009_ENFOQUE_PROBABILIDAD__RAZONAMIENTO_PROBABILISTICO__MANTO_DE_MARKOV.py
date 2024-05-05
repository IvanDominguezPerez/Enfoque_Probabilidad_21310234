import numpy as np

# Definimos la matriz de transición del manto de Markov
matriz_transicion = np.array([[0.8, 0.2],
                               [0.1, 0.9]])

# Definimos el estado inicial del manto de Markov
estado_inicial = np.array([0.5, 0.5])

# Simulamos el manto de Markov para 10 pasos de tiempo
num_pasos = 10
estado_actual = estado_inicial
for t in range(num_pasos):
    # Multiplicamos la matriz de transición por el estado actual para obtener el siguiente estado
    estado_siguiente = np.dot(estado_actual, matriz_transicion)
    # Actualizamos el estado actual para el siguiente paso de tiempo
    estado_actual = estado_siguiente

# Imprimimos el estado final después de los 10 pasos de tiempo
print("Estado final del manto de Markov después de", num_pasos, "pasos:", estado_actual)
