from hmmlearn import hmm
import numpy as np

class HiddenMarkovModel:
    def __init__(self, n_states, n_observations):
        """
        Inicializa el modelo de Markov oculto.

        Args:
        - n_states: Número de estados ocultos del modelo.
        - n_observations: Número de observaciones posibles.
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.model = None

    def fit(self, X):
        """
        Ajusta el modelo de Markov oculto a los datos de entrada.

        Args:
        - X: Lista de secuencias de observaciones.
        """
        # Crear el modelo HMM con el número de estados y observaciones especificados
        self.model = hmm.MultinomialHMM(n_components=self.n_states, n_iter=100)

        # Entrenar el modelo con las secuencias de observaciones
        self.model.fit(X)

    def predict(self, sequence):
        """
        Predice la secuencia de estados ocultos correspondiente a una secuencia de observaciones.

        Args:
        - sequence: Lista de observaciones.

        Returns:
        - hidden_states: Lista de estados ocultos predichos.
        """
        # Transformar la secuencia de observaciones en el formato esperado por el modelo
        X = np.atleast_2d(sequence).T

        # Realizar la predicción de los estados ocultos
        hidden_states = self.model.predict(X)

        return hidden_states
