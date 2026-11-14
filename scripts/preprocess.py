"""Corrige orientación EXIF, recorta a cuadrado y redimensiona UNA sola vez.

Requiere: python -m scripts.make_splits

Uso (desde la raíz del repositorio):
    python -m scripts.preprocess              # tamaño por defecto (config.PREPROCESS_SIZE)
    python -m scripts.preprocess --size 384
    python -m scripts.preprocess --force      # rehace aunque ya existan

Salida: data/processed/<size>/<misma ruta relativa que en el dataset>
        data/processed/<size>/manifest.json
Motivo: decodificar JPEG de ~9 MP en cada época es el cuello de botella en CPU.
"""
import argparse
import json
import sys

import pandas as pd
from tqdm import tqdm

from cacao import config
from cacao.data import find_dataset_root, load_square_image


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=config.PREPROCESS_SIZE)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    split_files = [config.SPLITS_DIR / f"{s}.csv" for s in ("train", "val", "test")]
    if not all(f.exists() for f in split_files):
        print("ERROR: faltan las particiones. Ejecuten: python -m scripts.make_splits")
        return 1
    df = pd.concat([pd.read_csv(f) for f in split_files], ignore_index=True)

    ds_root = find_dataset_root()
    out_root = config.PROCESSED_DIR / str(args.size)
    done, skipped = 0, 0
    for rel in tqdm(df["path"], desc=f"Preprocesando a {args.size}px"):
        dst = out_root / rel
        if dst.exists() and not args.force:
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        load_square_image(ds_root / rel, args.size).save(
            dst, "JPEG", quality=config.JPEG_QUALITY)
        done += 1

    missing = [r for r in df["path"] if not (out_root / r).exists()]
    total_mb = sum((out_root / r).stat().st_size for r in df["path"]
                   if (out_root / r).exists()) / 1e6
    manifest = {"size": args.size, "jpeg_quality": config.JPEG_QUALITY,
                "imagenes": len(df), "procesadas_ahora": done, "omitidas_ya_existian": skipped,
                "faltantes": len(missing), "tamano_total_mb": round(total_mb, 1)}
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
