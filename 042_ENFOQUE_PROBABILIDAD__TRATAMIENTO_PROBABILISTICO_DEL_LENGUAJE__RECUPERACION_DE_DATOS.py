import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Corpus de documentos de ejemplo
documents = [
    "La inteligencia artificial es emocionante",
    "La inteligencia artificial está revolucionando muchos campos",
    "Los algoritmos de inteligencia artificial son poderosos",
    "La robótica y la inteligencia artificial van de la mano",
    "La inteligencia artificial y el aprendizaje automático son áreas de investigación activa"
]

# Consulta de ejemplo
query = "¿Qué es la inteligencia artificial?"

# Construir la matriz de términos del documento
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# Convertir la consulta en un vector de términos
query_vector = vectorizer.transform([query])

# Calcular la similitud coseno entre la consulta y los documentos
similarity_scores = cosine_similarity(query_vector, X)

# Ordenar los documentos por similitud
sorted_indices = np.argsort(similarity_scores[0])[::-1]

# Imprimir los documentos ordenados por similitud con la consulta
print("Documentos ordenados por similitud con la consulta:")
for i, idx in enumerate(sorted_indices):
    print(f"{i+1}. {documents[idx]}")
