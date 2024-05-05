import nltk
from nltk import trigrams, bigrams
from collections import defaultdict

# Descargar el corpus de ejemplo (necesario si no se ha descargado previamente)
nltk.download('gutenberg')

# Cargar el corpus de ejemplo (Gutenberg)
corpus = nltk.corpus.gutenberg.sents('shakespeare-hamlet.txt')

# Preprocesamiento del corpus
corpus = [[word.lower() for word in sent if word.isalnum()] for sent in corpus]

# Construir modelo de trigramas
model = defaultdict(lambda: defaultdict(lambda: 0))
for sentence in corpus:
    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
        model[(w1, w2)][w3] += 1

# Convertir conteos a probabilidades
for prefix in model:
    total_count = float(sum(model[prefix].values()))
    for w3 in model[prefix]:
        model[prefix][w3] /= total_count

# Función para generar texto a partir del modelo
def generate_text(model, num_words=50, seed=None):
    text = []
    prefix = seed
    for _ in range(num_words):
        if prefix in model:
            w3 = max(model[prefix], key=model[prefix].get)
            text.append(w3)
            prefix = (prefix[1], w3)
        else:
            break
    return ' '.join(text)

# Generar texto de ejemplo a partir del modelo
generated_text = generate_text(model, seed=('the', 'king'))
print("Texto generado a partir del modelo:\n", generated_text)
