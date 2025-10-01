# tests/conftest.py
import sys
from pathlib import Path
import numpy as np
import pytest

# Añadir la carpeta raíz del proyecto al sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

@pytest.fixture
def imagen_rgb():
    h, w = 600, 800
    rng = np.random.default_rng(123)
    img = rng.integers(low=0, high=256, size=(h, w, 3), dtype=np.uint8)
    return img
