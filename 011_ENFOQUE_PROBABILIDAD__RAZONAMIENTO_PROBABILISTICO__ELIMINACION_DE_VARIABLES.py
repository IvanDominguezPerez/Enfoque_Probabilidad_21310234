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

# Creamos un objeto VariableElimination para realizar la eliminación de variables
eliminacion = VariableElimination(modelo)

# Realizamos eliminación de variables para calcular la probabilidad P(C)
resultado = eliminacion.query(variables=['C'])

# Imprimimos el resultado
print("Distribución marginal de C:")
print(resultado)
