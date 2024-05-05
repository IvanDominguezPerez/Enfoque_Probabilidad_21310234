import numpy as np
import matplotlib.pyplot as plt

# Definir una clase para el nodo en el árbol RRT
class Node:
    def __init__(self, config):
        self.config = config
        self.parent = None

# Definir la función para generar una configuración aleatoria en el espacio de configuración
def random_config():
    return np.random.rand(2) * 10  # Ejemplo: espacio de configuración 2D de 0 a 10 en ambos ejes

# Definir la función para encontrar el nodo más cercano en el árbol RRT
def nearest_node(nodes, config):
    distances = [np.linalg.norm(node.config - config) for node in nodes]
    nearest_index = np.argmin(distances)
    return nodes[nearest_index]

# Definir la función para expandir el árbol RRT hacia una nueva configuración
def expand_tree(nodes, target_config, step_size):
    nearest = nearest_node(nodes, target_config)
    direction = target_config - nearest.config
    distance = np.linalg.norm(direction)
    if distance > step_size:
        direction = direction / distance * step_size
    new_config = nearest.config + direction
    new_node = Node(new_config)
    new_node.parent = nearest
    return new_node

# Definir la función para buscar un camino desde la configuración inicial a la final utilizando RRT
def rrt(start_config, goal_config, num_iterations, step_size):
    # Crear el árbol RRT con el nodo inicial
    root = Node(start_config)
    nodes = [root]

    # Bucle de búsqueda de camino
    for _ in range(num_iterations):
        # Generar una configuración aleatoria
        target_config = random_config()

        # Expandir el árbol hacia la nueva configuración
        new_node = expand_tree(nodes, target_config, step_size)
        nodes.append(new_node)

        # Verificar si la nueva configuración está lo suficientemente cerca de la configuración objetivo
        if np.linalg.norm(new_node.config - goal_config) < step_size:
            goal_node = Node(goal_config)
            goal_node.parent = new_node
            nodes.append(goal_node)
            return nodes

    return None  # Si no se encuentra un camino dentro del número de iteraciones especificado

# Función para trazar el camino encontrado por RRT
def plot_rrt(nodes, start_config, goal_config):
    plt.figure(figsize=(8, 6))
    plt.plot(start_config[0], start_config[1], 'go', markersize=10, label='Inicio')
    plt.plot(goal_config[0], goal_config[1], 'ro', markersize=10, label='Objetivo')
    for node in nodes:
        if node.parent is not None:
            plt.plot([node.config[0], node.parent.config[0]], [node.config[1], node.parent.config[1]], 'k-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Camino encontrado por RRT')
    plt.legend()
    plt.grid(True)
    plt.show()

# Parámetros de la búsqueda de camino RRT
start_config = np.array([1, 1])    # Configuración inicial
goal_config = np.array([9, 9])     # Configuración objetivo
num_iterations = 500               # Número de iteraciones del algoritmo RRT
step_size = 0.5                    # Tamaño del paso para expandir el árbol RRT

# Ejecutar el algoritmo RRT para encontrar un camino desde la configuración inicial a la final
nodes = rrt(start_config, goal_config, num_iterations, step_size)

# Visualizar el camino encontrado por RRT
if nodes:
    plot_rrt(nodes, start_config, goal_config)
else:
    print("No se encontró un camino dentro del número especificado de iteraciones.")
