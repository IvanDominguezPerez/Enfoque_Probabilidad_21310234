import numpy as np

class GaussianMixtureModel:
    def __init__(self, num_components):
        """
        Inicializa el modelo de mezcla de gaussianas.

        Args:
        - num_components: Número de componentes de la mezcla.
        """
        self.num_components = num_components
        self.means = None
        self.covariances = None
        self.weights = None

    def fit(self, X, max_iters=100, tol=1e-4):
        """
        Estima los parámetros del modelo de mezcla de gaussianas utilizando el algoritmo EM.

        Args:
        - X: Matriz de observaciones (datos de entrenamiento).
        - max_iters: Número máximo de iteraciones.
        - tol: Tolerancia para la convergencia del algoritmo.
        """
        # Inicialización de parámetros
        N, D = X.shape
        self.means = X[np.random.choice(N, self.num_components, replace=False)]
        self.covariances = [np.eye(D) for _ in range(self.num_components)]
        self.weights = np.ones(self.num_components) / self.num_components

        prev_log_likelihood = float('-inf')
        for _ in range(max_iters):
            # Expectation step (E-step)
            responsibilities = self._calculate_responsibilities(X)

            # Maximization step (M-step)
            self._update_parameters(X, responsibilities)

            # Calcular log-verosimilitud y verificar convergencia
            log_likelihood = self._calculate_log_likelihood(X)
            if log_likelihood - prev_log_likelihood < tol:
                break
            prev_log_likelihood = log_likelihood

    def _calculate_responsibilities(self, X):
        """
        Calcula las responsabilidades de cada componente para cada observación.

        Args:
        - X: Matriz de observaciones (datos de entrenamiento).

        Returns:
        - Matriz de responsabilidades de tamaño (N, num_components).
        """
        N = X.shape[0]
        responsibilities = np.zeros((N, self.num_components))
        for k in range(self.num_components):
            # Calcular la probabilidad de cada observación bajo el k-ésimo componente
            probabilities = self._multivariate_normal_pdf(X, self.means[k], self.covariances[k])
            # Calcular la responsabilidad normalizada del k-ésimo componente para cada observación
            responsibilities[:, k] = self.weights[k] * probabilities
        # Normalizar las responsabilidades para que sumen 1 para cada observación
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        return responsibilities

    def _update_parameters(self, X, responsibilities):
        """
        Actualiza los parámetros del modelo utilizando las responsabilidades calculadas en el E-step.

        Args:
        - X: Matriz de observaciones (datos de entrenamiento).
        - responsibilities: Matriz de responsabilidades calculadas en el E-step.
        """
        N = X.shape[0]
        # Calcular la suma de las responsabilidades para cada componente
        component_sums = np.sum(responsibilities, axis=0)
        # Actualizar los pesos de los componentes
        self.weights = component_sums / N
        for k in range(self.num_components):
            # Actualizar la media del k-ésimo componente
            self.means[k] = np.sum(responsibilities[:, k, np.newaxis] * X, axis=0) / component_sums[k]
            # Calcular las matrices de covarianza del k-ésimo componente
            diffs = X - self.means[k]
            self.covariances[k] = np.dot((responsibilities[:, k, np.newaxis] * diffs).T, diffs) / component_sums[k]

    def _multivariate_normal_pdf(self, X, mean, covariance):
        """
        Calcula la función de densidad de probabilidad de una distribución normal multivariante.

        Args:
        - X: Matriz de observaciones.
        - mean: Vector de medias.
        - covariance: Matriz de covarianza.

        Returns:
        - Vector de probabilidades para cada observación.
        """
        D = X.shape[1]
        det_cov = np.linalg.det(covariance)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** D * det_cov)
        inv_cov = np.linalg
