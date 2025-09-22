# tests/test_preprocesar_predecir.py

import os
import numpy as np
import pytest
import importlib.util
from pathlib import Path

# Cargar el módulo principal
mod_path = Path(__file__).resolve().parents[1] / "detector_neumonia.py"
spec = importlib.util.spec_from_file_location("detector_neumonia", mod_path)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

def test_preprocesar_forma_tipo_rango(imagen_rgb):
    """
    Verifica que preprocess:
      - devuelva shape (1, 512, 512, 1)
      - dtype float32
      - valores en [0, 1]
    """
    salida = app.preprocess(imagen_rgb)
    assert isinstance(salida, np.ndarray)
    assert salida.shape == (1, 512, 512, 1)
    assert salida.dtype == np.float32
    assert np.min(salida) >= 0.0
    assert np.max(salida) <= 1.0


@pytest.mark.timeout(30)
def test_predecir_devuelve_resultado_valido(imagen_rgb):
    # Ruta al modelo junto al archivo detector_neumonia.py
    ruta_modelo = os.path.join(os.path.dirname(app.__file__), "conv_MLP_84.h5")

    # Si el modelo no existe o no está cargado, saltar prueba
    if not os.path.exists(ruta_modelo):
        pytest.skip("Modelo conv_MLP_84.h5 no encontrado; se omite esta prueba.")
    if not hasattr(app, "modelo"):
        pytest.skip("El modelo global 'modelo' no está definido en detector_neumonia.")

    etiqueta, probabilidad, mapa_calor = app.predict(imagen_rgb)

    assert isinstance(etiqueta, str)
    assert etiqueta in {"bacteriana", "normal", "viral"}
    assert isinstance(probabilidad, float)
    assert 0.0 <= probabilidad <= 100.0

    assert isinstance(mapa_calor, np.ndarray)
    assert mapa_calor.shape == imagen_rgb.shape
    assert mapa_calor.dtype == np.uint8