import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

class ProbabilisticLearning:
    def __init__(self, n_neighbors=5, n_clusters=3):
        """
        Inicializa el modelo de aprendizaje probabilístico.

        Args:
        - n_neighbors: Número de vecinos para el algoritmo k-NN.
        - n_clusters: Número de clusters para el algoritmo de K-Medias.
        """
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.knn_model = None
        self.kmeans_model = None

    def fit_knn(self, X_train, y_train):
        """
        Ajusta el modelo k-NN a los datos de entrenamiento.

        Args:
        - X_train: Datos de entrenamiento.
        - y_train: Etiquetas de clase correspondientes a los datos de entrenamiento.
        """
        # Inicializa el modelo k-NN con el número de vecinos especificado
        self.knn_model = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        # Ajusta el modelo k-NN a los datos de entrenamiento
        self.knn_model.fit(X_train, y_train)

    def predict_knn(self, X_test):
        """
        Realiza predicciones utilizando el modelo k-NN.

        Args:
        - X_test: Datos de prueba.

        Returns:
        - predictions: Predicciones realizadas por el modelo k-NN.
        """
        # Realiza predicciones utilizando el modelo k-NN entrenado
        predictions = self.knn_model.predict(X_test)

        return predictions

    def fit_kmeans(self, X):
        """
        Ajusta el modelo de K-Medias a los datos de entrada.

        Args:
        - X: Datos de entrada.
        """
        # Inicializa el modelo K-Medias con el número de clusters especificado
        self.kmeans_model = KMeans(n_clusters=self.n_clusters)

        # Ajusta el modelo K-Medias a los datos de entrada
        self.kmeans_model.fit(X)

    def predict_kmeans(self, X):
        """
        Asigna puntos de datos a los clusters identificados por el modelo K-Medias.

        Args:
        - X: Datos de entrada.

        Returns:
        - cluster_labels: Etiquetas de cluster asignadas a cada punto de datos.
        """
        # Asigna puntos de datos a los clusters identificados por el modelo K-Medias
        cluster_labels = self.kmeans_model.predict(X)

        return cluster_labels
