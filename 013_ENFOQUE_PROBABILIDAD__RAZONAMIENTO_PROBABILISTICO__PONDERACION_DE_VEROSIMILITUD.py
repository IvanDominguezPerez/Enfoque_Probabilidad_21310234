import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling

# Creamos un modelo bayesiano simple
modelo = BayesianModel([('A', 'C'), ('B', 'C')])

# Definimos las CPDs (Conditional Probability Distributions)
cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.6], [0.4]])
cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.7], [0.3]])
cpd_c = TabularCPD(variable='C', variable_card=2, 
                   values=[[0.8, 0.9, 0.6, 0.1], [0.2, 0.1, 0.4, 0.9]],
                   evidence=['A', 'B'], evidence_card=[2, 2])

# Añadimos las CPDs al modelo
modelo.add_cpds(cpd_a, cpd_b, cpd_c)

# Creamos un objeto BayesianModelSampling para realizar muestreo
muestreo = BayesianModelSampling(modelo)

# Realizamos muestreo directo para generar muestras de la distribución conjunta
muestras_directas = muestreo.forward_sample(size=1000)

# Definimos las observaciones parciales
observaciones = {'A': 0, 'B': 1}

# Calculamos los pesos de las muestras utilizando la ponderación de verosimilitud
pesos = np.prod([cpd_c.values[:, np.where(cpd_c.variables == 'C')[0][0]]
                 [muestras_directas['C'].values, observaciones['A'], observaciones['B']] 
                 for cpd_c in [cpd_c]], axis=0)

# Normalizamos los pesos para que sumen a 1
pesos_normalizados = pesos / np.sum(pesos)

# Imprimimos los pesos normalizados
print("Pesos normalizados de las muestras:")
print(pesos_normalizados)

