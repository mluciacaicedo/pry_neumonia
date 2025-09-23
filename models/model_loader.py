# model_loader.py
"""
Modulo encargado de cargar el modelo entrenado desde el archivo .h5
"""
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
def model_fun():
# cargar modelo desde el archivo .h5
    modelo = load_model(os.path.join(os.path.dirname(__file__), 'conv_MLP_84.h5'), compile=False)
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(reduction='sum_over_batch_size'),
        metrics=['accuracy']
    )
    return modelo
