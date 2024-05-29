import pandas as pd
import numpy as np

# Ruta del archivo CSV
file_path = 'datos/Salida_2024_05_29_16_17.csv'

# Leer el archivo CSV
data = pd.read_csv(file_path)

# Convertir las columnas 'start' y 'end' a tipo datetime con manejo de errores
data['start'] = pd.to_datetime(data['start'], errors='coerce')
data['end'] = pd.to_datetime(data['end'], errors='coerce')


data['high'] = pd.to_numeric(data['high'], errors='coerce')
data['low'] = pd.to_numeric(data['low'], errors='coerce')


# eliminar ceros y ausentes:
data[['vol','ask', 'bid', 'open', 'high', 'low', 'close']] = data[['vol','ask', 'bid', 'open', 'high', 'low', 'close']].replace(0, np.nan)
data.dropna(subset=['vol','ask', 'bid', 'open', 'high', 'low', 'close'], inplace=True)
data.dropna(subset=['start', 'end'], inplace=True)

# Calcular el promedio y la desviación estándar de la columna 'vol'
vol_mean = data['vol'].mean()
vol_std = data['vol'].std()

num_std = 3  # Por ejemplo, consideramos como límites 3 desviaciones estándar

# Filtrar los datos para mantener solo aquellos dentro del rango específico de desviaciones estándar del promedio
filtered_data = data[
                     (data['vol'] >= vol_mean - num_std * vol_std) & 
                     (data['vol'] <= vol_mean + num_std * vol_std)]


# Restablecer los índices después de eliminar filas
filtered_data = filtered_data.reset_index(drop=True)

filtered_data['trend'] = np.where(filtered_data['close'] >= filtered_data['open'], 1, -1)
filtered_data['d'] = abs(filtered_data['open'] - filtered_data['close'].shift(1))

# Calcular el promedio y la desviación estándar de la columna 'vol'
vol_mean = filtered_data['d'].mean()
vol_std = filtered_data['d'].std()

num_std = 3  # Por ejemplo, consideramos como límites 3 desviaciones estándar

# Filtrar los datos para mantener solo aquellos dentro del rango específico de desviaciones estándar del promedio
filtered_data = filtered_data[
                     (filtered_data['d'] >= vol_mean - num_std * vol_std) & 
                     (filtered_data['d'] <= vol_mean + num_std * vol_std)]


# Restablecer los índices después de eliminar filas
filtered_data = filtered_data.reset_index(drop=True)


filtered_data['condicion'] = np.nan

for i in range(1, len(filtered_data)):
    if i - 1 in filtered_data.index:
        # Tendencia anterior positiva
        if filtered_data.loc[i - 1, 'trend'] == 1:
            # open actual menor al close anterior
            if filtered_data.loc[i, 'open'] < filtered_data.loc[i - 1, 'close']:
                filtered_data.loc[i, 'condicion'] = 1
            else:
                filtered_data.loc[i, 'condicion'] = 0
        # Tendencia anterior negativa
        elif filtered_data.loc[i - 1, 'trend'] == -1:
            # open actual mayor al close anterior
            if filtered_data.loc[i, 'open'] > filtered_data.loc[i - 1, 'close']:
                filtered_data.loc[i, 'condicion'] = 1
            else:
                filtered_data.loc[i, 'condicion'] = 0


filtered_data['result'] = 0

for i in range(1, len(filtered_data)):
    # Verificar si la columna 'condicion' es igual a 1
    if filtered_data.loc[i, 'condicion'] == 1:
        # Verificar la tendencia de la fila anterior
        if filtered_data.loc[i - 1, 'trend'] == 1:
            # Verificar si el 'high' actual es mayor o igual al 'close' de la fila anterior
            if filtered_data.loc[i, 'high'] >= filtered_data.loc[i - 1, 'close']:
                filtered_data.loc[i, 'result'] = 1
        elif filtered_data.loc[i - 1, 'trend'] == -1:
            # Verificar si el 'low' actual es menor o igual al 'close' de la fila anterior
            if filtered_data.loc[i, 'low'] <= filtered_data.loc[i - 1, 'close']:
                filtered_data.loc[i, 'result'] = 1



# Mostrar el DataFrame con la nueva columna
print(filtered_data)

# Seleccionar solo las columnas 'open', 'close' y 'result'
final_data = filtered_data[['vol', 'ask', 'bid', 'open', 'close', 'high', 'low', 'result']]

# Guardar el DataFrame en un archivo CSV
output_file_path = 'datos/datos_class_01.csv'
final_data.to_csv(output_file_path, index=False)
