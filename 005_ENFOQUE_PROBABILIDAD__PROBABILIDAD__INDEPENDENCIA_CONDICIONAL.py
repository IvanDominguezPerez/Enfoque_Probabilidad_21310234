# Definimos la función de probabilidad conjunta P(A, B, C)
def probabilidad_conjunta(a, b, c):
    # Tabla de probabilidad conjunta (ejemplo)
    tabla_probabilidad = {
        ('A1', 'B1', 'C1'): 0.2,
        ('A1', 'B1', 'C2'): 0.1,
        ('A1', 'B2', 'C1'): 0.3,
        ('A1', 'B2', 'C2'): 0.1,
        ('A2', 'B1', 'C1'): 0.1,
        ('A2', 'B1', 'C2'): 0.2,
        ('A2', 'B2', 'C1'): 0.1,
        ('A2', 'B2', 'C2'): 0.1
    }
    return tabla_probabilidad.get((a, b, c), 0)

# Calculamos P(A|B, C)
def probabilidad_condicional_a_dado_b_y_c(a, b, c):
    # Calculamos P(A, B, C)
    prob_conjunta_abc = probabilidad_conjunta(a, b, c)
    # Calculamos P(B, C)
    prob_conjunta_bc = sum(probabilidad_conjunta(a, b, c) for a in {'A1', 'A2'})
    # Verificamos si P(B, C) es diferente de cero para evitar división por cero
    if prob_conjunta_bc == 0:
        return 0
    else:
        return prob_conjunta_abc / prob_conjunta_bc

# Ejemplo de uso
if __name__ == "__main__":
    # Definimos los eventos A, B y C
    evento_a = 'A1'
    evento_b = 'B1'
    evento_c = 'C1'

    # Calculamos P(A|B, C)
    probabilidad_condicional_a_dado_b_y_c_valor = probabilidad_condicional_a_dado_b_y_c(evento_a, evento_b, evento_c)
    print("Probabilidad condicional P(A|B, C):", probabilidad_condicional_a_dado_b_y_c_valor)
