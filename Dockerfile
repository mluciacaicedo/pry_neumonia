# Dockerfile definitivo sin GUI, funcional para tu proyecto

# Base
FROM python:3.12-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY . /app

# Instalar dependencias sin GUI
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.3.2 \
    opencv-python==4.9.0.80 \
    pillow==11.3.0 \
    pydicom==3.0.1 \
    scikit-image \
    tensorflow==2.16.1 \
    python-dotenv==1.1.1 \
    requests==2.32.5 \
    matplotlib \
    reportlab>=3.6.0

# Crear carpeta para resultados
RUN mkdir -p /app/reports

# Entrypoint directo a tu script
ENTRYPOINT ["python", "/app/src/detector_neumonia.py"]
