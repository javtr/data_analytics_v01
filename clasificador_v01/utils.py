import numpy as np
import random

def dividir_dataset(X, Y, test_size=0.2, random_state=0):
  """
  Función para dividir un dataset en conjuntos de entrenamiento y prueba, incluyendo índices originales.

  Args:
    X: Dataset con las variables o características.
    Y: Vector de etiquetas o valores que deseas predecir.
    test_size: Proporción de datos que se asignarán al conjunto de prueba (valor entre 0 y 1).
    random_state: Semilla aleatoria para controlar la aleatoriedad de la división.

  Returns:
    Tupla con seis conjuntos de datos: X_train, X_test, y_train, y_test, indices_originales, indices_originales_test.
  """

  # Número total de muestras
  num_muestras = len(X)

  # Número de muestras para el conjunto de prueba
  num_muestras_test = int(num_muestras * test_size)

  # Generar índices aleatorios
  indices_aleatorios = np.arange(num_muestras)
  np.random.seed(random_state)
  np.random.shuffle(indices_aleatorios)

  # Índices para el conjunto de entrenamiento
  indices_train = indices_aleatorios[:-num_muestras_test]

  # Índices para el conjunto de prueba
  indices_test = indices_aleatorios[-num_muestras_test:]

  # Índices originales
  indices_originales = np.arange(len(X))

  # Subconjuntos para entrenamiento y prueba
  X_train = X[indices_train]
  X_test = X[indices_test]
  y_train = Y[indices_train]
  y_test = Y[indices_test]

  # Índices originales para el conjunto de prueba
  indices_originales_test = indices_originales[indices_test]

  return X_train, X_test, y_train, y_test, indices_originales, indices_originales_test
