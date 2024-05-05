# Definimos las probabilidades condicionales P(A), P(B|A) y P(C|B)
probabilidad_a = 0.6
probabilidad_b_dado_a = 0.8
probabilidad_c_dado_b = 0.9

# Calculamos la probabilidad conjunta P(A, B, C) utilizando la regla de la cadena
probabilidad_conjunta = probabilidad_a * probabilidad_b_dado_a * probabilidad_c_dado_b

# Imprimimos el resultado
print("Probabilidad conjunta P(A, B, C):", probabilidad_conjunta)
