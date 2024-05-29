# Usar una imagen base de TensorFlow
FROM tensorflow/tensorflow:latest-gpu

# Crear un directorio de trabajo, nombre de la carpeta en host y en container
WORKDIR /app

# Install Python packages
RUN pip install pandas matplotlib scikit-learn

# Comando por defecto para ejecutar al iniciar el contenedor
CMD ["bash"]

