## Hola! Bienvenido a la herramienta para la detección rápida de neumonía

# Herramienta para la detección rápida de neumonía:

Aplicación basada en Deep Learning para el análisis de radiografías de tórax en formato DICOM/JPG/PNG con el fin de clasificarlas en tres categorías:

1. Neumonía Bacteriana

2. Neumonía Viral

3. Sin Neumonía

El sistema integra la técnica de interpretación Grad-CAM, que genera un mapa de calor resaltando las regiones más relevantes de la radiografía para la decisión de la red neuronal.

---

## Instalación y ejecución:
# Requisitos

1. Python 3.12
2. uv (recomendao) o pip
3. Librerías en requirements.txt
4. Modelo entrenado: conv_MLP_84.h5

---

## Pasos para la instalación:

Requerimientos necesarios para el funcionamiento:

# 1.Crear y activar entorno virtual
python -m venv .venv
# En Windows
.venv\Scripts\activate
# En macOS / Linux
source .venv/bin/activate

# 2. Instalar dependencias
uv pip sync requirements.txt
o si no se tiene uv 
pip install -r requirements.txt

# 3. Ejecutar aplicación 
python detector_neumonia.py 
ó  tambien usando el comando *uv run python detector_neumonia.py*

## Uso de la Interfaz Gráfica:
1. Ingrese la cédula del paciente en la caja de texto.
2. Presione el botón "Cargar Imagen" y seleccione la imagen (en formato .dcm, .jpg, .jpeg, .png).
3. Presione el botón "Predecir" y espere unos segundos, para obtener la clasificación y el mapa de calor.
4. Presión el botón "Borrar" para resetear y cargar nueva imagen.
5. Presione el botón "Guardar" para almacenar la información del paciente en excel con extensión .csv
6. Presione el botón "PDF" para descargar el resultado en PDF

las Imágenes de prueba, estan disponibles en:
https://drive.google.com/drive/folders/1PFLbGK8T95Mz2CyREFHSlpP7qha4uekV?usp=sharing


## Estructura del proyecto:

PRY_NEUMONIA/
│── models/
    └── model_loader.py
    └── conv_MLP_84.h5         # Es el modelo preen
│── reports/                   # carpeta donde se alamcenan los resultado obtenidos en las predicciones
│── src
    └── detector_neumonia.py       # Script principal, Contiene el diseño de la interfaz en Tkinter y lógica de predicción
    └── grad_camp.py     
    └── integrator.py
    └── predictor.py
    └── preprocess_img.py
    └── read_img.py                    
│── requirements.txt           # contiene las Dependencias
│── tests/                     # Carpeta de pruebas unitarias (pytest)
│    └── test_preprocesar_predecir.py

---
## Definiciones relevantes del proyecto: 

## integrator.py
Es un módulo que integra los demás scripts y retorna solamente lo necesario para ser visualizado en la interfaz gráfica. Retorna la clase, la probabilidad y una imagen el mapa de calor generado por Grad-CAM.

## read_img.py
Script que lee la imagen en formato DICOM para visualizarla en la interfaz gráfica. Además, la convierte a arreglo para su preprocesamiento.

## preprocess_img.py
Script que recibe el arreglo proveniento de read_img.py, realiza las siguientes modificaciones:

- resize a 512x512
- conversión a escala de grises
- ecualización del histograma con CLAHE
- normalización de la imagen entre 0 y 1
- conversión del arreglo de imagen a formato de batch (tensor)

## load_model.py
Script que lee el archivo binario del modelo de red neuronal convolucional previamente entrenado llamado 'conv_MLP_84.h5'.

## grad_cam.py
Script que recibe la imagen y la procesa, carga el modelo, obtiene la predicción y la capa convolucional de interés para obtener las características relevantes de la imagen.

---
## Librerías requeridas (Ver requirements.txt):

1. tensorflow 2.16.1: Framework principal para cargar y ejecutar el modelo .h5.
2. numpy 1.26.0: Para el  Procesamiento numérico y manejo de tensores.
3. opencv-python 4.9.0: para el Procesamiento de imágenes (lectura, conversión a escala de grises, resize).
4. pillow 10.2.0 para el manejo de imágenes (Tkinter y generación de reportes).
5. pydicom 2.4.3: para la Lectura de imágenes médicas en formato DICOM.
6. tkcap 0.0.1: para la captura de pantalla de la interfaz Tkinter para generar PDF.
7. pytest 8.0.0: es el Framework para ejecutar pruebas unitarias.

---
## Arquitectura del modelo
La red neuronal convolucional implementada (CNN) es basada en el modelo implementado por F. Pasa, V.Golkov, F. Pfeifer, D. Cremers & D. Pfeifer
en su artículo Efcient Deep Network Architectures for Fast Chest X-Ray Tuberculosis Screening and Visualization.

Está compuesta por 5 bloques convolucionales, cada uno contiene 3 convoluciones; dos secuenciales y una conexión 'skip' que evita el desvanecimiento del gradiente a medida que se avanza en profundidad.
Con 16, 32, 48, 64 y 80 filtros de 3x3 para cada bloque respectivamente.

Después de cada bloque convolucional se encuentra una capa de max pooling y después de la última una capa de Average Pooling seguida por tres capas fully-connected (Dense) de 1024, 1024 y 3 neuronas respectivamente.

Para regularizar el modelo utilizamos 3 capas de Dropout al 20%; dos en los bloques 4 y 5 conv y otra después de la 1ra capa Dense.

## Acerca de Grad-CAM
Es una técnica utilizada para resaltar las regiones de una imagen que son importantes para la clasificación. Un mapeo de activaciones de clase para una categoría en particular indica las regiones de imagen relevantes utilizadas por la CNN para identificar esa categoría.

Grad-CAM realiza el cálculo del gradiente de la salida correspondiente a la clase a visualizar con respecto a las neuronas de una cierta capa de la CNN. Esto permite tener información de la importancia de cada neurona en el proceso de decisión de esa clase en particular. Una vez obtenidos estos pesos, se realiza una combinación lineal entre el mapa de activaciones de la capa y los pesos, de esta manera, se captura la importancia del mapa de activaciones para la clase en particular y se ve reflejado en la imagen de entrada como un mapa de calor con intensidades más altas en aquellas regiones relevantes para la red con las que clasificó la imagen en cierta categoría.



---
## Pruebas unitarias implementadas con pytest
El proyecto incluye pruebas automáticas para validar las funciones críticas, las cuales puedes ejecutar mediante el comando: *pytest -v*

A continuacion te mostramos las unitarias implementadas:

# 1. Prueba para validar que el preprocesamiento de imágenes funciona correctamente:
En esta prueba comprobamos que la función de preprocesamiento (preprocess) realmente prepara la imagen como Tensor para el modelo.

- validamos que la salida de preprocess sea un numpy.ndarray.
- validamos que el tamaño final sea correcto: (1, 512, 512, 1) (batch con 1 imagen, 512x512 píxeles, 1 canal).
- validamos que los valores estén normalizados entre 0.0 y 1.0.
- validamos que el tipo de dato sea float32, el estándar para TensorFlow/Keras.

# 2. Prueba para validar que el modelo devuelve predicciones válidas:
En esta prueba comprobamos que el modelo devuelve predicciones válidas (etiqueta, probabilidad y heatmap).

- validamos que la función predict() devuelva exactamente una tupla (label, proba, heatmap).
- validamos que la etiqueta (label) sea un str válido y pertenezca al conjunto "bacteriana", "normal", "viral".
- validamos que la probabilidad (proba) sea un float y esté en el rango 0.0 – 100.0.
- validamos que el heatmap sea un numpy.ndarray, tenga el mismo tamaño que la imagen de entrada y en formato uint8.
- Incluye un timeout de 30s para evitar que la prueba se cuelgue si el modelo tarda demasiado.


----

## Créditos

# Proyecto original realizado por:
Isabella Torres Revelo → https://github.com/isa-tr
Nicolas Diaz Salazar → https://github.com/nicolasdiazsalazar

# Adaptaciones y mejoras en esta versión:
- Compatibilidad con TensorFlow 2.16 y Python 3.12
- Correcciones en Grad-CAM con tf.GradientTape
- Manejo unificado para imágenes DICOM/JPG/PNG
- Inclusión de pruebas unitarias con pytest
- Documentación técnica y guía de uso mejorada

