import numpy as np

class BayesianLinearRegression:
    def __init__(self, alpha, beta):
        """
        Inicializa un modelo de regresión lineal bayesiano.

        Args:
        - alpha: Parámetro de precisión del prior sobre los pesos.
        - beta: Parámetro de precisión del likelihood.
        """
        self.alpha = alpha
        self.beta = beta
        self.mean = None
        self.covariance = None

    def fit(self, X, y):
        """
        Entrena el modelo de regresión lineal bayesiano.

        Args:
        - X: Matriz de características (observaciones).
        - y: Vector de etiquetas (valores objetivo).
        """
        # Obtener dimensiones de los datos
        N, D = X.shape

        # Calcular la matriz de diseño Phi
        Phi = np.hstack((np.ones((N, 1)), X))

        # Actualizar la media y covarianza del posterior
        self.covariance = np.linalg.inv(self.alpha * np.eye(D + 1) + self.beta * Phi.T @ Phi)
        self.mean = self.beta * self.covariance @ Phi.T @ y

    def predict(self, X):
        """
        Realiza predicciones utilizando el modelo de regresión lineal bayesiano.

        Args:
        - X: Matriz de características (observaciones).

        Returns:
        - Predicciones para las observaciones dadas.
        """
        # Calcular la matriz de diseño Phi
        Phi = np.hstack((np.ones((X.shape[0], 1)), X))

        # Calcular la media predictiva y la varianza predictiva
        mean_pred = Phi @ self.mean
        variance_pred = 1 / self.beta + np.sum(Phi @ self.covariance * Phi, axis=1)

        return mean_pred, variance_pred

# Datos de ejemplo
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])

# Hiperparámetros alpha y beta (hiperparámetros del prior)
alpha = 1
beta = 1

# Creamos una instancia del modelo de regresión lineal bayesiano
model = BayesianLinearRegression(alpha, beta)

# Entrenamos el modelo
model.fit(X_train, y_train)

# Realizamos predicciones para nuevos datos
X_test = np.array([[6], [7]])
mean_pred, variance_pred = model.predict(X_test)

# Imprimimos las predicciones y las varianzas predictivas
for i in range(len(X_test)):
    print("Predicción para x =", X_test[i], ": media =", mean_pred[i], ", varianza =", variance_pred[i])
