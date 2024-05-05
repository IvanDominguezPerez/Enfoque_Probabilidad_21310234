from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

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

# Creamos un objeto VariableElimination para realizar inferencia por enumeración
inferencia = VariableElimination(modelo)

# Realizamos inferencia para calcular la probabilidad P(C|A=0, B=1)
resultado = inferencia.query(variables=['C'], evidence={'A': 0, 'B': 1})

# Imprimimos el resultado
print("Probabilidad posterior de C dado A=0 y B=1:")
print(resultado)
