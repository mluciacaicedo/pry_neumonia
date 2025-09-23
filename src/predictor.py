""
Este módulo se encarga de realizar la prediccion sobre una imagen medica, tomando la imagen como un arreglo tipo numpy y la envia
el modulo de preprocesamiento
""
# predictor.py
import sys
import os
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
from model_loader import model_fun
from grad_cam import grad_cam
from preprocess_img import preprocess

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
    heatmap = grad_cam(array, batch_array_img)
    return (label, proba, heatmap)
