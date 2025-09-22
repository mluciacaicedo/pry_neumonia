# integrator.py
#Este un módulo que integra los demás scripts y retorna solamente lo necesario para ser visualizado en la interfaz

from read_img import read_dicom_file, read_jpg_file
from predictor import predict

def run_pipeline(filepath):
    if filepath.lower().endswith(".dcm"):
        array, _ = read_dicom_file(filepath)
    else:
        array, _ = read_jpg_file(filepath)
    label, proba, heatmap = predict(array)
    return label, proba, heatmap