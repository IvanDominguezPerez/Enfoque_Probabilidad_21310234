# Definimos la probabilidad a priori de un evento
probabilidad_a_priori = 0.3

# Definimos la función de probabilidad condicional (likelihood)
def probabilidad_condicional(evento, hipotesis):
    if hipotesis == 'H1':  # Suponiendo dos hipótesis posibles
        if evento == 'E1':
            return 0.8
        else:
            return 0.2
    elif hipotesis == 'H2':
        if evento == 'E1':
            return 0.4
        else:
            return 0.6
    else:
        raise ValueError("Hipótesis no válida")

# Calculamos la probabilidad a priori para cada hipótesis
probabilidad_a_priori_H1 = probabilidad_a_priori
probabilidad_a_priori_H2 = 1 - probabilidad_a_priori

# Evento observado
evento_observado = 'E1'

# Calculamos la probabilidad total del evento observado sumando sobre todas las hipótesis
probabilidad_total_evento = (probabilidad_condicional(evento_observado, 'H1') * probabilidad_a_priori_H1) + \
                            (probabilidad_condicional(evento_observado, 'H2') * probabilidad_a_priori_H2)

# Calculamos la probabilidad a posteriori utilizando el teorema de Bayes
probabilidad_a_posteriori_H1 = (probabilidad_condicional(evento_observado, 'H1') * probabilidad_a_priori_H1) / \
                                probabilidad_total_evento
probabilidad_a_posteriori_H2 = (probabilidad_condicional(evento_observado, 'H2') * probabilidad_a_priori_H2) / \
                                probabilidad_total_evento

# Imprimimos los resultados
print("Probabilidad a priori de la hipótesis H1:", probabilidad_a_priori_H1)
print("Probabilidad a priori de la hipótesis H2:", probabilidad_a_priori_H2)
print("Probabilidad a posteriori de la hipótesis H1 después de observar el evento", evento_observado, 
      ":", probabilidad_a_posteriori_H1)
print("Probabilidad a posteriori de la hipótesis H2 después de observar el evento", evento_observado, 
      ":", probabilidad_a_posteriori_H2)
