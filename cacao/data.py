"""Carga y transformación de imágenes compartida por scripts y app.

`load_square_image` es la ÚNICA función que convierte una foto en la entrada del modelo.
La usan preprocess.py (entrenamiento) y streamlit_app.py (demo): así ambas etapas
aplican exactamente la misma orientación, recorte y redimensionado.
"""
from pathlib import Path

from PIL import Image, ImageOps

from cacao import config


def find_dataset_root(raw_dir: Path = config.RAW_DIR) -> Path:
    """Carpeta que contiene cocoapods_images/ dentro de data/raw/."""
    roots = sorted(p.parent for p in raw_dir.rglob("cocoapods_images") if p.is_dir())
    if not roots:
        raise FileNotFoundError("No se encontró 'cocoapods_images/'. Ejecuten antes "
                                "python -m scripts.download_data")
    if len(roots) > 1:
        print(f"AVISO: varias copias del dataset; se usa {roots[0]}")
    return roots[0]


def load_square_image(src, size: int) -> Image.Image:
    """Abre una imagen (ruta o archivo subido), corrige la orientación EXIF,
    recorta al centro a un cuadrado y redimensiona a size x size (RGB).

    - EXIF: el 84,5 % de las fotos del dataset trae rotación en metadatos.
    - Recorte central: las fotos del dataset son cuadradas (el recorte no cambia nada);
      las del usuario pueden ser 4:3 o 16:9 y así no se deforman.
    """
    with Image.open(src) as im:
        im.draft("RGB", (size, size))   # JPEG: decodifica a escala reducida (>= size)
        im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    im = im.crop((left, top, left + s, top + s))
    return im.resize((size, size), Image.Resampling.LANCZOS)
