"""Configuración central del proyecto.

Todas las rutas y constantes viven aquí para que los scripts y la app
usen exactamente los mismos valores (evita diferencias entre entrenamiento y demo).
"""
from pathlib import Path

# --- Rutas (relativas a la raíz del repositorio) ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Dataset: CocoaMoniliaDataSet (CC BY 4.0) ---
# Versión 2, citada en el artículo de Data in Brief (DOI 10.1016/j.dib.2025.112447)
ZENODO_RECORD_ID = "17716661"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
# Respaldo: URL de la versión 1 usada en el notebook oficial del repositorio de los autores
ZENODO_FALLBACK_V1_URL = (
    "https://zenodo.org/records/17156052/files/CocoaMoniliaDataSet.zip?download=1"
)

# --- Clases (nombres de carpeta del dataset) ---
CLASSES = ["h0", "m1", "m2", "m3"]
CLASS_NAMES = {
    "h0": "Sana",
    "m1": "Monilia ciclo 1 (jorobas)",
    "m2": "Monilia ciclos 2-3 (manchas marrones/aceitosas)",
    "m3": "Monilia ciclo 4 (esporulación)",
}
EXPECTED_TOTAL_IMAGES = 1953  # según el artículo; se verifica tras la descarga

# --- Reproducibilidad y modelado ---
SEED = 42
IMG_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
