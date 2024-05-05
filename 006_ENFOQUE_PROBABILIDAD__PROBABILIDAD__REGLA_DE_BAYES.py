# Definimos la función de probabilidad condicional P(A|B)
def probabilidad_condicional_a_dado_b(probabilidad_b_dado_a, probabilidad_a, probabilidad_b):
    # Aplicamos la regla de Bayes
    return (probabilidad_b_dado_a * probabilidad_a) / probabilidad_b

# Ejemplo de uso
if __name__ == "__main__":
    # Definimos la probabilidad condicional P(B|A)
    probabilidad_b_dado_a = 0.8
    # Definimos la probabilidad a priori P(A)
    probabilidad_a = 0.6
    # Definimos la probabilidad a priori P(B)
    probabilidad_b = 0.7

    # Calculamos la probabilidad condicional P(A|B)
    probabilidad_condicional_a_dado_b_valor = probabilidad_condicional_a_dado_b(probabilidad_b_dado_a, probabilidad_a, probabilidad_b)
    print("Probabilidad condicional P(A|B):", probabilidad_condicional_a_dado_b_valor)
