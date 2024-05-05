import matplotlib.pyplot as plt

# Datos de ejemplo
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Crear el gráfico
plt.figure(figsize=(8, 6))  # Definir el tamaño del gráfico
plt.plot(x, y, marker='o', linestyle='-', color='b', label='Datos de ejemplo')  # Graficar los datos
plt.xlabel('Eje X')  # Etiqueta del eje X
plt.ylabel('Eje Y')  # Etiqueta del eje Y
plt.title('Gráfico de ejemplo')  # Título del gráfico
plt.grid(True)  # Mostrar rejilla en el gráfico
plt.legend()  # Mostrar leyenda
plt.show()  # Mostrar el gráfico
