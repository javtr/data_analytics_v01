import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from utils import dividir_dataset
import time

# List all available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("----------- GPUs disponibles:", gpus)

# Configura TensorFlow para permitir el crecimiento de la memoria GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

start_time = time.time()




# ============ 1. Importar y arreglar datos ====================================

# # Importar el dataset de entrenamiento
# dataset_train = pd.read_csv("MAR24.csv")

# importar datos
directorio = "datos"
dataset_train = pd.DataFrame()

for archivo in os.listdir(directorio):
    if archivo.endswith('.csv'):
        df = pd.read_csv(os.path.join(directorio, archivo), header=None)
        dataset_train = pd.concat([dataset_train, df], ignore_index=True)


# Renombrar columnas
columnas_nuevas = [
    'swing', 'resultado', 'v0', 'd0', 'b0', 'v1', 'd1', 'y1', 'b1', 'v2', 'd2', 'y2', 'b2', 'v3', 'd3', 'y3', 'b3', 'v4', 'd4', 'y4', 'b4', 'v5', 'd5', 'y5', 'b5'
]

# Verifica si la cantidad de columnas coincide
assert len(dataset_train.columns) == len(
    columnas_nuevas), "La cantidad de columnas no coincide"
dataset_train.columns = columnas_nuevas

# ordenar los datos
dataset_train_sorted = dataset_train.sort_values(by='swing')



# Balancear el numero de resultados para evitar sesgo
proporcion = 0

# Bucle para equilibrar mientras la proporción sea menor al 90%
while proporcion < 0.9:
    # Contar el número de ejemplos para cada clase
    n_cero = dataset_train_sorted[dataset_train_sorted['resultado'] == 0].shape[0]
    n_mayor_cero = dataset_train_sorted[dataset_train_sorted['resultado'] > 0].shape[0]

    # Calcular la proporción entre las clases
    proporcion = n_cero / n_mayor_cero

    # Estrategia de equilibrio
    if n_mayor_cero > n_cero:
        # Submuestreo
        n_filas_eliminar = n_mayor_cero - n_cero
        filas_eliminar = np.random.choice(
            dataset_train_sorted[dataset_train_sorted['resultado'] > 0].index, size=n_filas_eliminar)
        dataset_train_sorted = dataset_train_sorted.drop(
            filas_eliminar, axis=0)
    else:
        # Sobremuestreo
        n_filas_copiar = n_cero - n_mayor_cero
        filas_copiar = dataset_train_sorted[dataset_train_sorted['resultado'] == 0].sample(
            n=n_filas_copiar, replace=True)
        dataset_train_sorted = dataset_train_sorted.append(
            filas_copiar, ignore_index=True)

# Verificar la proporción final
n_cero_final = dataset_train_sorted[dataset_train_sorted['resultado'] == 0].shape[0]
n_mayor_cero_final = dataset_train_sorted[dataset_train_sorted['resultado'] > 0].shape[0]

proporcion_final = n_cero_final / n_mayor_cero_final

print('-------')
print('0:', n_cero_final)
print('1:', n_mayor_cero_final)
print('Proporción:', proporcion_final)

# 13:2 nodos, 9:1 nodo
# Separar variables de entrada con resultado
# X = dataset_train_sorted.iloc[:, 2:9].values
# Y = dataset_train_sorted.iloc[:, 1:2].values

columnas_entrada = ['v0', 'd0', 'b0', 'v1', 'd1', 'y1', 'b1', 'v2', 'd2', 'y2', 'b2', 'v3', 'd3', 'y3', 'b3', 'v4', 'd4', 'y4', 'b4', 'v5', 'd5', 'y5', 'b5']
columnas_resultado = ['resultado']

X = dataset_train_sorted[columnas_entrada].values
Y = dataset_train_sorted[columnas_resultado].values

# pasar los datos a binario
Y_binary = np.where(Y > 0, 1, Y)

# Dividir el dataset entre train y test
# X_train, X_test, y_train, y_test = train_test_split(
#     X, Y_binary, test_size=0.2, random_state=0)


X_train, X_test, y_train, y_test, indices_originales, indices_originales_test = dividir_dataset(X, Y, test_size=0.2)

# pasar los datos a binario
y_train =  np.where(y_train > 0, 1, y_train)
y_test =  np.where(y_test > 0, 1, y_test)

# Escalar los datos de entrada y salida
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# ============ 2. Construir la RNA ====================================


classifier = Sequential([
    Dense(1024, kernel_initializer="LecunUniform",
          activation='leaky_relu', input_dim=len(columnas_entrada)),
    Dense(512, kernel_initializer="LecunUniform", activation='leaky_relu'),
    Dense(256, kernel_initializer="LecunUniform", activation='leaky_relu'),
    Dense(128, kernel_initializer="LecunUniform", activation='leaky_relu'),
    Dense(1, kernel_initializer="LecunUniform", activation="sigmoid")
])

classifier.compile(
    optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=100, verbose=1)

# ============ 3. Prediccion ====================================
y_pred = classifier.predict(X_test)
y_pred_porcentajes = y_pred

# porcentaje de aceptacion de resultado, si es mayor al % se toma como 1
y_pred_pos = (y_pred > 0.9)
y_pred_neg = (y_pred > 0.1)

# ============ 4. Evaluar el resultado ====================================

# 1. matriz de confusion
# esto separa las precisiones entre positivas y negativas
# muestra que tan bueno es prediciendo en cada clase
# muestra que tan equilibrada esta la respuesta entre negativa y positiva comparando con los reales
'''
[VN][FP]
[FN][VP]
Precision= V/V+F
'''
cm = confusion_matrix(y_test, y_pred_pos)

print("Matriz de Precision Positiva ===================================")
print("[" + str(cm[0][0]) + ", " + str(cm[0][1]) +
      "]\n[" + str(cm[1][0]) + ", " + str(cm[1][1]) + "]")
print("--------")
print("Precision Negativa: " + str(round(cm[0][0] / (cm[0][0] + cm[1][0]), 2)))
print("Precision Positiva: " + str(round(cm[1][1] / (cm[0][1] + cm[1][1]), 2)))
print("Precision Total: " + str(round((cm[0][0] + cm[1][1]) / cm.sum(), 2)))
print("--------")
print("Total Negativa Prediccion: " + str(round((cm[0][0] + cm[1][0]), 2)))
print("Total Positiva Prediccion: " + str(round((cm[0][1] + cm[1][1]), 2)))
print("--------")
print("Total Negativa Test: " + str(np.sum(y_test == 0)))
print("Total Positiva Test: " + str(np.sum(y_test == 1)))


# Precision Negativa
cmn = confusion_matrix(y_test, y_pred_neg)

print("Matriz de Precision Negativa ===================================")
print("[" + str(cmn[0][0]) + ", " + str(cmn[0][1]) +
      "]\n[" + str(cmn[1][0]) + ", " + str(cmn[1][1]) + "]")
print("--------")
print("Precision Negativa: " + str(round(cmn[0][0] / (cmn[0][0] + cmn[1][0]), 2)))
print("Precision Positiva: " + str(round(cmn[1][1] / (cmn[0][1] + cmn[1][1]), 2)))
print("Precision Total: " + str(round((cmn[0][0] + cmn[1][1]) / cmn.sum(), 2)))
print("--------")
print("Total Negativa Prediccion: " + str(round((cmn[0][0] + cmn[1][0]), 2)))
print("Total Positiva Prediccion: " + str(round((cmn[0][1] + cmn[1][1]), 2)))
print("--------")
print("Total Negativa Test: " + str(np.sum(y_test == 0)))
print("Total Positiva Test: " + str(np.sum(y_test == 1)))



# 2. distribucion normal de los porcentajes predichos
# esto indica que tan fuerte es el patron encontrado
plt.hist(y_pred_porcentajes, bins=10,
         edgecolor='black', linewidth=1, zorder=1000)
plt.grid(True, which='both', linestyle='--',
         linewidth=0.5, color='gray', zorder=-1)
plt.xlabel('Porcentaje')
plt.ylabel('Frecuencia')
plt.title('Distribución de porcentajes de predicción')

# Ajustar las etiquetas del eje x para que se muestren cada 0.1
plt.xticks([i/10 for i in range(11)]) 

# plt.show()


# 3. Obtener casos correctos
'''
# obtener los indices predichos correctamente
indices_correctos = np.where(y_pred == y_test)[0]

# relacionar índices con los originales 
indices_coincidentes = indices_originales_test[indices_correctos]

# obtener filas de valores correctos
filas_coincidentes = dataset_train_sorted[dataset_train_sorted.index.isin(indices_coincidentes)]
'''

# ============ 5. Validar la confiabilidad del resultado ====================================
"""

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score


def build_classifier():

    classifier = Sequential([
        Dense(256, kernel_initializer="LecunUniform", activation='relu', input_dim=17),
        Dense(128, kernel_initializer="LecunUniform", activation='relu', kernel_regularizer=l2(0.0001)),
        Dense(64, kernel_initializer="LecunUniform",activation='relu', kernel_regularizer=l2(0.0001)),
        Dense(1, kernel_initializer="LecunUniform",activation="sigmoid")
    ])

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
    
    return classifier

# Configurar el clasificador Keras
classifier = KerasClassifier(model=build_classifier,epochs=100,loss="binary_crossentropy")

# Realizar validación cruzada
accuracies = cross_val_score(classifier, X=X_train, y=y_train, cv=10, n_jobs=-1, verbose=1, scoring='accuracy')

print("Cross-validation accuracies:", accuracies)

"""

print("End")

total_time = time.time()
print(f"Tiempo total: {total_time - start_time:.2f} segundos")

