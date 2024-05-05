from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import matplotlib.pyplot as plt

# Creamos un objeto BayesianModel
modelo = BayesianModel([('A', 'C'), ('B', 'C')])

# Creamos las Tabular Conditional Probability Distributions (CPDs)
cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.6], [0.4]])
cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.7], [0.3]])
cpd_c = TabularCPD(variable='C', variable_card=2, 
                   values=[[0.8, 0.9, 0.6, 0.1], [0.2, 0.1, 0.4, 0.9]],
                   evidence=['A', 'B'], evidence_card=[2, 2])

# Añadimos las CPDs al modelo
modelo.add_cpds(cpd_a, cpd_b, cpd_c)

# Verificamos si el modelo es válido
print("El modelo es válido:", modelo.check_model())

# Dibujamos el grafo de la red bayesiana
nx.draw(modelo, with_labels=True, node_color='lightblue', node_size=2000, font_size=15, font_weight='bold')
plt.title("Red Bayesiana")
plt.show()
