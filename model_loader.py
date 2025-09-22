#Usa una variable privada _model_instance para mantener el modelo cargado en memoria.

# model_loader.py
import tensorflow as tf
from tensorflow.keras.models import load_model

# cargar modelo desde el archivo .h5 (mismo comportamiento que tu script)
modelo = load_model('conv_MLP_84.h5', compile=False)
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(reduction='sum_over_batch_size'),
    metrics=['accuracy']
)

def model_fun():
    return modelo