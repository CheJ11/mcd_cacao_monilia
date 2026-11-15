"""Construye la galería de ejemplos de la app con imágenes del conjunto de TEST.

Requiere: python -m scripts.evaluate --models mobilenetv2 --split test

Uso (desde la raíz del repositorio):
    python -m scripts.make_gallery              # 3 aciertos + 1 error por clase
    python -m scripts.make_gallery --per-class 5 --errors 2

Criterio: imágenes que el modelo nunca vio al entrenar. Se incluyen errores reales a
propósito para mostrar honestamente los límites del modelo. Selección aleatoria con
semilla fija (reproducible).

Salida: app_assets/gallery/ejemplo_XX.jpg y app_assets/gallery/gallery.json
Licencia de las imágenes: CC BY 4.0 (CocoaMoniliaDataSet); la app muestra la atribución.
"""
import argparse
import json
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image

from cacao import config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--errors", type=int, default=1, help="errores reales por clase (si existen)")
    ap.add_argument("--quality", type=int, default=85)
    args = ap.parse_args()

    pred_f = config.REPORTS_DIR / f"predictions_{config.APP_MODEL}_test.csv"
    if not pred_f.exists():
        print(f"ERROR: falta {pred_f}. Ejecuten la evaluación en test primero.")
        return 1
    pred = pd.read_csv(pred_f)
    rng = np.random.default_rng(config.SEED)

    picks = []
    for c in config.CLASSES:
        sub = pred[pred["cls"] == c]
        bad = sub[~sub["correcto"]]
        ok = sub[sub["correcto"]]
        n_bad = min(args.errors, len(bad))
        idx_bad = rng.permutation(len(bad))[:n_bad]
        idx_ok = rng.permutation(len(ok))[:args.per_class - n_bad]
        picks += [bad.iloc[i] for i in idx_bad] + [ok.iloc[i] for i in idx_ok]
    order = rng.permutation(len(picks))  # mezcladas: el público no ve la clase de antemano

    if config.GALLERY_DIR.exists():
        shutil.rmtree(config.GALLERY_DIR)
    config.GALLERY_DIR.mkdir(parents=True)
    src_root = config.PROCESSED_DIR / str(config.PREPROCESS_SIZE)
    items = []
    for n, i in enumerate(order, start=1):
        r = picks[i]
        fname = f"ejemplo_{n:02d}.jpg"
        Image.open(src_root / r["path"]).convert("RGB").save(
            config.GALLERY_DIR / fname, "JPEG", quality=args.quality)
        items.append({"archivo": fname, "clase_real": r["cls"],
                      "origen": r["path"], "error_en_evaluacion": bool(not r["correcto"])})
    (config.GALLERY_DIR / "gallery.json").write_text(
        json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    size_mb = sum(p.stat().st_size for p in config.GALLERY_DIR.glob("*.jpg")) / 1e6
    print(f"{len(items)} imágenes en {config.GALLERY_DIR} ({size_mb:.1f} MB)")
    print(pd.DataFrame(items).groupby("clase_real")["error_en_evaluacion"]
          .agg(total="size", errores="sum").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
