from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo (se generan aleatoriamente)
np.random.seed(0)
data = np.random.rand(100, 2)

# Definir el tamaño del mapa SOM
som_size = (10, 10)  # 10x10

# Inicializar y entrenar el SOM
som = MiniSom(som_size[0], som_size[1], 2, sigma=0.5, learning_rate=0.5)
som.train_random(data, 1000)  # 1000 iteraciones de entrenamiento

# Visualizar el mapa SOM y los datos originales
plt.figure(figsize=(8, 8))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # visualizar la matriz de distancias
plt.colorbar()

# Marcar las neuronas en el mapa
for i, j in np.ndindex(som_size):
    plt.text(i + 0.5, j + 0.5, '{}'.format(som.winner(np.array([i, j]))), ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))

# Marcar los datos originales en el mapa SOM
for idx, d in enumerate(data):
    winner = som.winner(d)
    plt.plot(winner[0] + 0.5, winner[1] + 0.5, 'o', markeredgecolor='k', markerfacecolor='None')
    if idx < len(data) - 1:
        next_winner = som.winner(data[idx + 1])
        plt.arrow(winner[0] + 0.5, winner[1] + 0.5, next_winner[0] - winner[0], next_winner[1] - winner[1],
                  head_width=0.1, head_length=0.1, fc='r', ec='r')

plt.title('Mapa Autoorganizado de Kohonen')
plt.show()
