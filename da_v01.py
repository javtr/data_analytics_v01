import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo de gráficos con un tema oscuro personalizado
custom_dark_style = {
    'axes.facecolor': '#1f1f1f',    # Color de fondo de los ejes
    'axes.edgecolor': '#ffffff',     # Color del borde de los ejes
    'axes.labelcolor': '#ffffff',    # Color de las etiquetas de los ejes
    'xtick.color': '#ffffff',        # Color de las marcas del eje x
    'ytick.color': '#ffffff',        # Color de las marcas del eje y
    'text.color': '#ffffff',         # Color del texto
    'figure.facecolor': '#121212',   # Color de fondo de la figura
    'grid.color': '#555555',         # Color de las líneas de la cuadrícula
}

plt.style.use(custom_dark_style)

# Ruta del archivo CSV
file_path = 'datos/export_2024_05_22_17_21.csv'

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






## --- graficas ----------------------------------------------------


'''
# Crear una figura y un conjunto de subtramas (subplots) con cuatro subvistas
fig, axs = plt.subplots(2, 2, figsize=(14, 14))  # 2x2 subplots

# Graficar el volumen de trading a lo largo del tiempo en la primera subvista
axs[0, 0].plot(data['end'], data['vol'], label='Volumen', color='blue')  # Cambia el color aquí
axs[0, 0].set_xlabel('Fecha')
axs[0, 0].set_ylabel('Volumen')
axs[0, 0].set_title('Volumen de trading a lo largo del tiempo')
axs[0, 0].legend()

# Graficar el precio de cierre a lo largo del tiempo en la segunda subvista
axs[0, 1].plot(data['end'], data['close'], label='Precio de cierre', color='red')  # Cambia el color aquí
axs[0, 1].set_xlabel('Fecha')
axs[0, 1].set_ylabel('Precio de cierre')
axs[0, 1].set_title('Precio de cierre a lo largo del tiempo')
axs[0, 1].legend()

# Graficar el precio de apertura a lo largo del tiempo en la tercera subvista
axs[1, 0].plot(data['end'], data['open'], label='Precio de apertura', color='green')  # Cambia el color aquí
axs[1, 0].set_xlabel('Fecha')
axs[1, 0].set_ylabel('Precio de apertura')
axs[1, 0].set_title('Precio de apertura a lo largo del tiempo')
axs[1, 0].legend()

# Graficar la matriz de correlación en la cuarta subvista
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axs[1, 1])
axs[1, 1].set_title('Matriz de correlación')

# Ajustar el layout para que no se sobrepongan las subtramas
plt.tight_layout()

# Mostrar todas las gráficas al final
plt.show()

'''