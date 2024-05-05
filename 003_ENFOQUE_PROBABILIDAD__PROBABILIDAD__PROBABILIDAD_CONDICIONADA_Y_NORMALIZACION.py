# Definimos la función de probabilidad conjunta P(X, Y)
def probabilidad_conjunta(x, y):
    # Tabla de probabilidad conjunta (ejemplo)
    tabla_probabilidad = {
        ('X1', 'Y1'): 0.2,
        ('X1', 'Y2'): 0.3,
        ('X2', 'Y1'): 0.1,
        ('X2', 'Y2'): 0.4
    }
    return tabla_probabilidad.get((x, y), 0)

# Definimos la función de probabilidad marginal P(X)
def probabilidad_marginal_x(x):
    # Sumamos sobre todas las posibles observaciones de Y
    return sum(probabilidad_conjunta(x, y) for y in {'Y1', 'Y2'})

# Calculamos la probabilidad condicionada P(Y|X)
def probabilidad_condicionada_y_dado_x(y, x):
    probabilidad_conjunta_xy = probabilidad_conjunta(x, y)
    probabilidad_marginal_x_valor = probabilidad_marginal_x(x)
    if probabilidad_marginal_x_valor == 0:
        return 0
    else:
        return probabilidad_conjunta_xy / probabilidad_marginal_x_valor

# Normalizamos las probabilidades
def normalizar_probabilidades(probabilidades):
    total = sum(probabilidades.values())
    return {clave: valor / total for clave, valor in probabilidades.items()}

# Calculamos la probabilidad condicionada P(X|Y)
def probabilidad_condicionada_x_dado_y(x, y):
    # Calculamos P(X, Y) y P(Y)
    probabilidad_conjunta_xy = probabilidad_conjunta(x, y)
    probabilidad_marginal_y = sum(probabilidad_conjunta(x, y) for x in {'X1', 'X2'})
    # Calculamos P(X|Y) utilizando el teorema de Bayes
    if probabilidad_marginal_y == 0:
        return 0
    else:
        return probabilidad_conjunta_xy / probabilidad_marginal_y

# Ejemplo de uso
if __name__ == "__main__":
    # Calculamos P(Y|X)
    y_dado_x = {}
    for y in {'Y1', 'Y2'}:
        for x in {'X1', 'X2'}:
            y_dado_x[(y, x)] = probabilidad_condicionada_y_dado_x(y, x)
    print("Probabilidad condicionada P(Y|X):", y_dado_x)

    # Normalizamos las probabilidades condicionadas P(Y|X)
    y_dado_x_normalizado = normalizar_probabilidades(y_dado_x)
    print("Probabilidad condicionada normalizada P(Y|X):", y_dado_x_normalizado)

    # Calculamos P(X|Y)
    x_dado_y = {}
    for x in {'X1', 'X2'}:
        for y in {'Y1', 'Y2'}:
            x_dado_y[(x, y)] = probabilidad_condicionada_x_dado_y(x, y)
    print("Probabilidad condicionada P(X|Y):", x_dado_y)

    # Normalizamos las probabilidades condicionadas P(X|Y)
    x_dado_y_normalizado = normalizar_probabilidades(x_dado_y)
    print("Probabilidad condicionada normalizada P(X|Y):", x_dado_y_normalizado)
