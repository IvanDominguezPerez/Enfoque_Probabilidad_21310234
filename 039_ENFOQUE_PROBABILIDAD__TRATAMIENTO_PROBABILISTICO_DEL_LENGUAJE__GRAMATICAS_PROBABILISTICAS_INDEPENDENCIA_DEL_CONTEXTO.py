import nltk
import random

# Definición de la gramática probabilística de independencia de contexto
grammar = nltk.PCFG.fromstring("""
    S -> NP VP [1.0]
    NP -> Det N [0.5] | NP PP [0.3] | 'John' [0.2]
    Det -> 'the' [0.6] | 'a' [0.4]
    N -> 'man' [0.5] | 'dog' [0.3] | 'cat' [0.2]
    VP -> V NP [0.6] | VP PP [0.4]
    V -> 'chased' [0.5] | 'saw' [0.3] | 'ate' [0.2]
    PP -> P NP [1.0]
    P -> 'with' [0.6] | 'in' [0.4]
""")

# Crear un generador de texto basado en la PCFG
def generate_text(grammar, start_symbol='S'):
    # Crear un parser de la gramática
    parser = nltk.ChartParser(grammar)
    # Generar un árbol de parseo válido
    trees = list(parser.parse(start_symbol))
    # Elegir aleatoriamente un árbol de parseo
    tree = random.choice(trees)
    # Obtener la lista de palabras del árbol de parseo
    words = tree.leaves()
    # Unir las palabras en una sola cadena de texto
    return ' '.join(words)

# Generar una frase utilizando la gramática probabilística
generated_sentence = generate_text(grammar)
print("Frase generada:", generated_sentence)
