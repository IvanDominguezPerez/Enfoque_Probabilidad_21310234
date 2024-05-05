import nltk
import numpy as np

# Corpus paralelo de ejemplo (inglés-español)
english_corpus = ["I like to eat apples", "She loves playing soccer"]
spanish_corpus = ["Me gusta comer manzanas", "A ella le encanta jugar al fútbol"]

# Tokenización y creación de trigramas para ambos corpus
english_tokens = [nltk.word_tokenize(sent.lower()) for sent in english_corpus]
spanish_tokens = [nltk.word_tokenize(sent.lower()) for sent in spanish_corpus]

english_trigrams = [trigram for sent in english_tokens for trigram in nltk.trigrams(sent)]
spanish_trigrams = [trigram for sent in spanish_tokens for trigram in nltk.trigrams(sent)]

# Crear un diccionario que mapea trigramas inglés-español
translation_dict = {}
for en_tri, es_tri in zip(english_trigrams, spanish_trigrams):
    translation_dict[en_tri] = es_tri

# Función para traducir una oración de inglés a español utilizando el modelo de trigramas
def translate_sentence(sentence, translation_dict):
    translated_sentence = []
    for en_tri in nltk.trigrams(sentence.lower().split()):
        if en_tri in translation_dict:
            translated_sentence.extend(translation_dict[en_tri])
        else:
            translated_sentence.append("UNK")  # Palabra desconocida si no se encuentra en el diccionario
    return " ".join(translated_sentence)

# Ejemplo de traducción de una oración de inglés a español
input_sentence = "I like soccer"
translated_sentence = translate_sentence(input_sentence, translation_dict)
print("Traducción de la oración:", translated_sentence)
