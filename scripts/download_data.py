"""Descarga, verifica (MD5) y descomprime CocoaMoniliaDataSet desde Zenodo.

Uso (desde la raíz del repositorio):
    python -m scripts.download_data            # descarga y descomprime
    python -m scripts.download_data --list     # solo lista los archivos del registro
    python -m scripts.download_data --fallback-v1  # usa la versión 1 si la API falla

Es idempotente: si el archivo ya existe y su MD5 coincide, no lo descarga de nuevo.
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from cacao import config

CHUNK = 1024 * 1024  # 1 MB


def get_record_files(api_url: str) -> list[dict]:
    """Devuelve [{'key', 'size', 'md5', 'url'}] a partir de la API de Zenodo."""
    resp = requests.get(api_url, timeout=60)
    resp.raise_for_status()
    files = []
    for f in resp.json().get("files", []):
        checksum = f.get("checksum", "")
        files.append({
            "key": f["key"],
            "size": f.get("size"),
            "md5": checksum.split(":", 1)[1] if checksum.startswith("md5:") else None,
            "url": f["links"]["self"],
        })
    return files


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, size: int | None = None) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = size or int(r.headers.get("content-length", 0)) or None
        with open(tmp, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True,
                                         desc=dest.name) as bar:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)  # solo se renombra si la descarga terminó completa


def safe_extract(zip_path: Path, target: Path) -> None:
    """Descomprime evitando rutas fuera de `target` (protección zip-slip)."""
    target = target.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            out = (target / member.filename).resolve()
            if target not in out.parents and out != target:
                raise RuntimeError(f"Ruta insegura en el zip: {member.filename}")
        zf.extractall(target)


def count_images(raw_dir: Path) -> dict[str, int]:
    """Cuenta imágenes por clase dentro de cualquier carpeta cocoapods_images/."""
    counts = {c: 0 for c in config.CLASSES}
    for img_root in raw_dir.rglob("cocoapods_images"):
        for c in config.CLASSES:
            d = img_root / c
            if d.is_dir():
                counts[c] += sum(1 for p in d.iterdir()
                                 if p.suffix.lower() in config.IMAGE_EXTENSIONS)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="solo listar archivos del registro")
    ap.add_argument("--fallback-v1", action="store_true",
                    help="descargar la versión 1 (URL del notebook oficial)")
    args = ap.parse_args()

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.fallback_v1:
        print("AVISO: usando la versión 1 del dataset (no la citada en el artículo).")
        files = [{"key": "CocoaMoniliaDataSet_v1.zip", "size": None, "md5": None,
                  "url": config.ZENODO_FALLBACK_V1_URL}]
    else:
        print(f"Consultando {config.ZENODO_API_URL}")
        files = get_record_files(config.ZENODO_API_URL)
        for f in files:
            mb = f["size"] / 1e6 if f["size"] else float("nan")
            print(f"  - {f['key']}  ({mb:,.1f} MB)  md5={f['md5']}")
        if args.list:
            return 0
        files = [f for f in files if f["key"].lower().endswith(".zip")]
        if not files:
            print("ERROR: el registro no contiene archivos .zip; revisen la lista anterior.")
            return 1

    for f in files:
        dest = config.RAW_DIR / f["key"]
        if dest.exists() and f["md5"] and md5sum(dest) == f["md5"]:
            print(f"OK (ya descargado y verificado): {dest.name}")
        else:
            download(f["url"], dest, f["size"])
            if f["md5"]:
                got = md5sum(dest)
                if got != f["md5"]:
                    print(f"ERROR: MD5 no coincide para {dest.name} ({got} != {f['md5']})")
                    return 1
                print(f"MD5 verificado: {dest.name}")
        print(f"Descomprimiendo {dest.name} ...")
        safe_extract(dest, config.RAW_DIR)

    counts = count_images(config.RAW_DIR)
    total = sum(counts.values())
    print("\nImágenes por clase:", json.dumps(counts))
    print(f"Total: {total} (esperado según el artículo: {config.EXPECTED_TOTAL_IMAGES})")
    if total != config.EXPECTED_TOTAL_IMAGES:
        print("AVISO: el total difiere del artículo; se analizará en el EDA.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
