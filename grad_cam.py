#Contiene la lógica para generar la visualización Grad-CAM (Class Activation Maps) a partir de una imagen y un modelo ya cargado.

# grad_cam.py
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K 
from preprocess_img import preprocess
from model_loader import model_fun

def grad_cam(array):
    img = preprocess(array)
    model = model_fun()
    preds = model.predict(img)
    argmax = np.argmax(preds[0])
    last_conv_layer = model.get_layer("conv10_thisone")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, pred_out = grad_model(img)
        # Ajuste para manejar salidas envueltas y poder indexar
        if isinstance(pred_out, (list, tuple)):
            pred_out = pred_out[0]
        elif isinstance(pred_out, dict):
            pred_out = next(iter(pred_out.values()))
        pred_out = tf.convert_to_tensor(pred_out)
        loss = pred_out[:, argmax]

    grads = tape.gradient(loss, conv_out)
    pooled_grads_value = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_layer_output_value = conv_out[0].numpy()

    for i in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # creating the heatmap
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap = cv2.resize(heatmap, (array.shape[1], array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img2 = array  # mantener tamaño del array original
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2)
    superimposed_img = superimposed_img.astype(np.uint8)
    return superimposed_img[:, :, ::-1]
