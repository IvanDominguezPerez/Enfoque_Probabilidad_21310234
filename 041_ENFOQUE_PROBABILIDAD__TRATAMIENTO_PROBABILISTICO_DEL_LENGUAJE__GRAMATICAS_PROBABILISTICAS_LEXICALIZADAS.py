import nltk
import random

# Definición de la gramática probabilística lexicalizada
grammar = nltk.PCFG.fromstring("""
    S -> NP VP [1.0]
    NP -> Det N [0.6] | 'John' [0.2] | 'I' [0.2]
    VP -> V NP [0.7] | VP PP [0.3]
    PP -> P NP [1.0]
    Det -> 'the' [0.8] | 'my' [0.2]
    N -> 'man' [0.5] | 'dog' [0.3] | 'cat' [0.2]
    V -> 'chased' [0.5] | 'saw' [0.3] | 'ate' [0.2]
    P -> 'with' [0.6] | 'in' [0.4]
""")

# Crear un generador de texto basado en la gramática
def generate_text(grammar, start_symbol='S'):
    # Crear un parser de la gramática
    parser = nltk.ViterbiParser(grammar)
    # Generar un árbol de parseo válido
    tree = list(parser.parse(start_symbol))[0].copy()
    # Obtener la lista de palabras del árbol de parseo
    words = tree.leaves()
    # Unir las palabras en una sola cadena de texto
    return ' '.join(words)

# Generar una frase utilizando la gramática probabilística lexicalizada
generated_sentence = generate_text(grammar)
print("Frase generada:", generated_sentence)
