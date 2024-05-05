import numpy as np

class NaiveBayes:
    def __init__(self):
        """
        Inicializa el clasificador Naive Bayes.
        """
        self.class_priors = None
        self.feature_likelihoods = None

    def fit(self, X, y):
        """
        Entrena el clasificador Naive Bayes.

        Args:
        - X: Matriz de características (observaciones).
        - y: Vector de etiquetas de clase.
        """
        # Obtener dimensiones de los datos
        N, D = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)

        # Calcular probabilidades a priori de clase
        self.class_priors = np.zeros(num_classes)
        for i, c in enumerate(self.classes):
            self.class_priors[i] = np.sum(y == c) / N

        # Inicializar matriz de likelihoods de características
        self.feature_likelihoods = np.zeros((num_classes, D))

        # Calcular likelihoods de características para cada clase
        for i, c in enumerate(self.classes):
            # Obtener índices de instancias de clase c
            indices = np.where(y == c)
            # Calcular likelihoods de características para clase c
            self.feature_likelihoods[i] = np.mean(X[indices], axis=0)

    def predict(self, X):
        """
        Realiza predicciones utilizando el clasificador Naive Bayes.

        Args:
        - X: Matriz de características (observaciones).

        Returns:
        - Vector de etiquetas de clase predichas.
        """
        # Calcular probabilidades posteriores para cada clase
        posteriors = np.zeros((X.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes):
            # Calcular log-probabilidades de características para clase c
            log_likelihoods = np.sum(np.log(self.feature_likelihoods[i])) * X + \
                              np.sum(np.log(1 - self.feature_likelihoods[i])) * (1 - X)
            # Calcular log-probabilidades posteriores
            posteriors[:, i] = np.log(self.class_priors[i]) + log_likelihoods

        # Predecir la clase con la probabilidad posterior más alta
        predictions = np.argmax(posteriors, axis=1)
        return predictions

# Datos de ejemplo (reseñas de películas IMDB)
X_train = np.array([[1, 0, 1, 1], [1, 1, 1, 0], [0, 1, 0, 1], [1, 1, 0, 1]])
y_train = np.array([1, 1, 0, 0])  # Etiquetas de clase (1: positiva, 0: negativa)

# Creamos una instancia del clasificador Naive Bayes
model = NaiveBayes()

# Entrenamos el modelo
model.fit(X_train, y_train)

# Realizamos predicciones para nuevas observaciones
X_test = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
predictions = model.predict(X_test)

# Imprimimos las predicciones
print("Predicciones:", predictions)
