#Este módulo se encargará de Tomar una imagen como np.array, y llamar al módulo preprocess_img.py para preparar la imagen.

# predictor.py
import numpy as np
from model_loader import model_fun
from preprocess_img import preprocess
from grad_cam import grad_cam

def predict(array):
    # preprocesamiento
    batch_array_img = preprocess(array)
    # cargar modelo y predecir
    model = model_fun()
    preds = model.predict(batch_array_img)
    prediction = np.argmax(preds)
    proba = float(np.max(preds) * 100)
    label = ""
    if prediction == 0:
        label = "bacteriana"
    if prediction == 1:
        label = "normal"
    if prediction == 2:
        label = "viral"
    # Grad-CAM
    heatmap = grad_cam(array)
    return (label, proba, heatmap)
