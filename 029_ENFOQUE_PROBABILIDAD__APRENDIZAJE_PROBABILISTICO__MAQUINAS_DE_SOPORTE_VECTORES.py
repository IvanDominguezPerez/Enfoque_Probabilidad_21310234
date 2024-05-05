from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class ProbabilisticLearning:
    def __init__(self, kernel='rbf', C=1.0):
        """
        Inicializa el modelo de aprendizaje probabilístico.

        Args:
        - kernel: Tipo de núcleo para SVM. Puede ser 'linear', 'poly', 'rbf', 'sigmoid', entre otros.
        - C: Parámetro de regularización.
        """
        self.kernel = kernel
        self.C = C
        self.svm_model = None

    def fit_svm(self, X_train, y_train):
        """
        Ajusta el modelo SVM a los datos de entrenamiento.

        Args:
        - X_train: Datos de entrenamiento.
        - y_train: Etiquetas de clase correspondientes a los datos de entrenamiento.
        """
        # Inicializa el modelo SVM con el kernel y parámetro C especificados
        self.svm_model = SVC(kernel=self.kernel, C=self.C)

        # Ajusta el modelo SVM a los datos de entrenamiento
        self.svm_model.fit(X_train, y_train)

    def predict_svm(self, X_test):
        """
        Realiza predicciones utilizando el modelo SVM.

        Args:
        - X_test: Datos de prueba.

        Returns:
        - predictions: Predicciones realizadas por el modelo SVM.
        """
        # Realiza predicciones utilizando el modelo SVM entrenado
        predictions = self.svm_model.predict(X_test)

        return predictions

# Genera datos de ejemplo para clasificación binaria
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Divide los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea una instancia de la clase ProbabilisticLearning
svm_model = ProbabilisticLearning(kernel='rbf', C=1.0)

# Ajusta el modelo SVM a los datos de entrenamiento
svm_model.fit_svm(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
predictions = svm_model.predict_svm(X_test)

# Calcula la precisión de las predicciones
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
