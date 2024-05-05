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

# Realizamos muestreo por rechazo para generar muestras de la distribución condicional P(C|A=0, B=1)
muestras_rechazo = muestreo.rejection_sample(evidence={'A': 0, 'B': 1}, size=1000)

# Imprimimos las muestras obtenidas
print("Muestras obtenidas mediante muestreo directo:")
print(muestras_directas.head())
print("\nMuestras obtenidas mediante muestreo por rechazo:")
print(muestras_rechazo.head())
