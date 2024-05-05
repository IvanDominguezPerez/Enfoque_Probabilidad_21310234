import spacy
import re

# Cargar el modelo de procesamiento del lenguaje de spaCy (en inglés)
nlp = spacy.load("en_core_web_sm")

# Texto de ejemplo
text = """
El lanzamiento de SpaceX destruyó un satélite de Facebook en órbita. 
El satélite, llamado Amos-6, estaba destinado a proporcionar conectividad de Internet a África.
"""

# Procesar el texto con spaCy
doc = nlp(text)

# Definir patrones de expresiones regulares para identificar nombres de organizaciones y entidades con números
org_pattern = re.compile(r'([A-Z][a-z]*(\s+[A-Z][a-z]*)*([^\w\s]|$))')
number_pattern = re.compile(r'\d{1,2}\s?[A-Z]{1,2}[a-z]{2,9}\s?\d{2,4}')

# Extraer información
organizations = set()  # Almacenar nombres de organizaciones únicas
dates = set()          # Almacenar fechas únicas
for entity in doc.ents:
    if entity.label_ == "ORG":  # Si la entidad es una organización
        match = org_pattern.search(entity.text)
        if match:
            organizations.add(match.group(0))
    elif entity.label_ == "DATE":  # Si la entidad es una fecha
        dates.add(entity.text)
    elif entity.label_ == "CARDINAL":  # Si la entidad es un número cardinal (posiblemente una fecha)
        match = number_pattern.search(entity.text)
        if match:
            dates.add(match.group(0))

# Imprimir las organizaciones identificadas
print("Organizaciones:")
for org in organizations:
    print(org)

# Imprimir las fechas identificadas
print("\nFechas:")
for date in dates:
    print(date)
