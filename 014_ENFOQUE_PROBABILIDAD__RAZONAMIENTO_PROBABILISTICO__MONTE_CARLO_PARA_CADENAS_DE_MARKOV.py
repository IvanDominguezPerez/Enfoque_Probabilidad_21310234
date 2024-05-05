import numpy as np
import pymc3 as pm

# Definimos la matriz de transición de la cadena de Markov
P = np.array([[0.9, 0.1],
              [0.2, 0.8]])

# Definimos la distribución de probabilidad inicial
pi_0 = np.array([0.5, 0.5])

# Número de iteraciones
n_iter = 10000

# Generamos muestras utilizando el método de Monte Carlo para Cadenas de Markov
with pm.Model() as modelo_mcmc:
    # Definimos la variable de estado de la cadena de Markov
    estado = pm.Categorical('estado', p=pi_0)
    
    # Iteramos para generar muestras
    for i in range(n_iter):
        # Transición de estado según la matriz de transición
        estado_next = pm.Categorical('estado_{}'.format(i), p=P[estado])
        estado = estado_next
        
    # Realizamos el muestreo utilizando el algoritmo Metropolis-Hastings
    trace = pm.sample(n_iter, tune=0, chains=1)
