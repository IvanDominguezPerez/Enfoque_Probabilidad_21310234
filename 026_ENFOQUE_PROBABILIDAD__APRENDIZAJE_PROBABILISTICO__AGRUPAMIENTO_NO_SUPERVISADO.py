import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iters=300, tol=1e-4):
        """
        Inicializa el modelo K-Means.

        Args:
        - n_clusters: Número de clusters a buscar.
        - max_iters: Número máximo de iteraciones.
        - tol: Tolerancia para la convergencia del algoritmo.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        """
        Ajusta el modelo K-Means a los datos de entrada.

        Args:
        - X: Matriz de datos de entrada.
        """
        # Inicialización aleatoria de los centroides
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iters):
            # Asignación de puntos al cluster más cercano
            clusters = self._assign_clusters(X)

            # Actualización de los centroides
            new_centroids = self._update_centroids(X, clusters)

            # Verificar convergencia
            if np.allclose(new_centroids, self.centroids, atol=self.tol):
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        """
        Asigna cada punto de los datos de entrada al cluster más cercano.

        Args:
        - X: Matriz de datos de entrada.

        Returns:
        - clusters: Lista de clusters con los índices de los puntos asignados a cada uno.
        """
        clusters = [[] for _ in range(self.n_clusters)]

        for i, x in enumerate(X):
            # Calcular la distancia de cada punto a todos los centroides
            distances = [np.linalg.norm(x - centroid) for centroid in self.centroids]
            # Asignar el punto al cluster con la distancia mínima
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(i)

        return clusters

    def _update_centroids(self, X, clusters):
        """
        Actualiza los centroides basados en la media de los puntos asignados a cada cluster.

        Args:
        - X: Matriz de datos de entrada.
        - clusters: Lista de clusters con los índices de los puntos asignados a cada uno.

        Returns:
        - new_centroids: Nuevos centroides actualizados.
        """
        new_centroids = np.zeros_like(self.centroids)

        for i, cluster in enumerate(clusters):
            if len(cluster) > 0:
                # Calcular la media de los puntos asignados al cluster
                new_centroids[i] = np.mean(X[cluster], axis=0)

        return new_centroids
