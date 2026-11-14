"""Análisis exploratorio (EDA) de CocoaMoniliaDataSet.

Uso (desde la raíz del repositorio):
    python -m scripts.eda                 # análisis completo
    python -m scripts.eda --threshold 4   # umbral de Hamming para casi-duplicados

Salidas:
    reports/eda_summary.json            resumen numérico de todo el análisis
    reports/images_metadata.csv         una fila por imagen (clase, tamaño, EXIF, split oficial)
    reports/near_duplicates.csv         pares candidatos (pHash, distancia <= 12) con indicador
    reports/repeated_filenames.csv      archivos con el mismo nombre en varias carpetas de clase
    reports/figures/eda_*.png           figuras para las diapositivas
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import imagehash
import matplotlib
matplotlib.use("Agg")  # sin ventanas: el script solo guarda figuras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from cacao import config

EXIF_ORIENTATION = 274  # etiqueta EXIF estándar de orientación


# ----------------------------------------------------------------- localización
def find_dataset_root(raw_dir: Path) -> Path:
    roots = sorted(p.parent for p in raw_dir.rglob("cocoapods_images") if p.is_dir())
    if not roots:
        raise FileNotFoundError("No se encontró 'cocoapods_images/'. Ejecuten antes "
                                "python -m scripts.download_data")
    if len(roots) > 1:
        print(f"AVISO: varias copias del dataset; se usa {roots[0]}")
    return roots[0]


# ------------------------------------------------------------------ 1. inventario
def inventory(ds_root: Path) -> dict:
    """Archivos por carpeta de primer nivel y extensión (explica diferencias de conteo)."""
    inv: dict[str, Counter] = {}
    for p in ds_root.rglob("*"):
        if p.is_file():
            top = p.relative_to(ds_root).parts[0]
            inv.setdefault(top, Counter())[p.suffix.lower() or "(sin ext.)"] += 1
    return {k: dict(v) for k, v in sorted(inv.items())}


def non_image_files_in_classes(ds_root: Path) -> list[str]:
    img_root = ds_root / "cocoapods_images"
    return [str(p.relative_to(ds_root)) for p in img_root.rglob("*")
            if p.is_file() and p.suffix.lower() not in config.IMAGE_EXTENSIONS]


# ---------------------------------------------------------- 2. metadatos por imagen
def image_metadata(ds_root: Path) -> pd.DataFrame:
    rows = []
    paths = [(c, p) for c in config.CLASSES
             for p in sorted((ds_root / "cocoapods_images" / c).iterdir())
             if p.suffix.lower() in config.IMAGE_EXTENSIONS]
    for cls, p in tqdm(paths, desc="Metadatos + pHash"):
        with Image.open(p) as im:
            w, h = im.size                      # tamaño "crudo", sin aplicar EXIF
            orient = im.getexif().get(EXIF_ORIENTATION, 1)
            im.draft("RGB", (512, 512))         # decodificación JPEG reducida: mucho más rápida
            im = ImageOps.exif_transpose(im)    # misma orientación que verá el modelo
            phash = int(str(imagehash.phash(im)), 16)
        rows.append({"file": p.name, "cls": cls, "path": str(p.relative_to(ds_root)),
                     "width": w, "height": h, "exif_orientation": orient,
                     "size_mb": p.stat().st_size / 1e6, "phash": phash})
    return pd.DataFrame(rows)


# --------------------------------------------------------- 3. anotaciones COCO
def coco_index(ds_root: Path) -> tuple[dict, dict]:
    """Une los 4 JSON COCO y deduplica anotaciones por (imagen, categoría, caja).

    Devuelve:
      idx:  {nombre_archivo: {"cats": Counter(categoría->n_mazorcas), "rel_areas": [...]}}
      stats: resumen por JSON y global.
    """
    idx: dict[str, dict] = {}
    seen: set = set()
    stats = {"por_json": {}}
    for cls in config.CLASSES:
        f = ds_root / "COCO_annotations" / f"instances_{cls}.json"
        if not f.exists():
            stats["por_json"][cls] = "no encontrado"
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        imgs = {i["id"]: i for i in d.get("images", [])}
        cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        stats["por_json"][cls] = {"imagenes": len(imgs),
                                  "anotaciones": len(d.get("annotations", [])),
                                  "categorias_definidas": sorted(set(cats.values()))}
        for im in imgs.values():
            idx.setdefault(Path(im["file_name"]).name, {"cats": Counter(), "rel_areas": []})
        for a in d.get("annotations", []):
            im = imgs.get(a["image_id"])
            if im is None or not a.get("bbox"):
                continue
            name, cat = Path(im["file_name"]).name, cats.get(a["category_id"])
            key = (name, cat, tuple(round(v, 1) for v in a["bbox"]))
            if key in seen:  # la misma mazorca aparece en varios JSON
                continue
            seen.add(key)
            idx[name]["cats"][cat] += 1
            if im.get("width") and im.get("height"):
                idx[name]["rel_areas"].append(a["bbox"][2] * a["bbox"][3]
                                              / (im["width"] * im["height"]))
    by_cat = Counter()
    for v in idx.values():
        by_cat.update(v["cats"])
    n_pods = np.array([sum(v["cats"].values()) for v in idx.values()])
    areas = [a for v in idx.values() for a in v["rel_areas"]]
    stats["global"] = {
        "imagenes_unicas": len(idx),
        "mazorcas_unicas": int(n_pods.sum()),
        "mazorcas_por_categoria": dict(sorted(by_cat.items())),
        "imagenes_por_n_mazorcas": {int(k): int(v) for k, v in
                                    zip(*np.unique(n_pods, return_counts=True))},
        "imagenes_con_categorias_mezcladas": sum(len(v["cats"]) > 1 for v in idx.values()),
        "area_bbox_relativa_mediana": float(np.median(areas)) if areas else None,
    }
    return idx, stats


def label_quality(df: pd.DataFrame, idx: dict) -> dict:
    """Agrega a df la información COCO por imagen y resume la consistencia de etiquetas."""
    info = [idx.get(n) for n in df["file"]]
    df["n_pods"] = [sum(i["cats"].values()) if i else 0 for i in info]
    df["coco_cats"] = ["|".join(sorted(i["cats"])) if i else "" for i in info]
    df["n_folders"] = df.groupby("file")["cls"].transform("nunique")
    df["label_ok"] = [(c == cls) for c, cls in zip(df["coco_cats"], df["cls"])]
    df["clean_single_pod"] = df["label_ok"] & (df["n_pods"] == 1) & (df["n_folders"] == 1)
    folder_names = set(df["file"])
    return {
        "imagenes_coco_sin_archivo": sorted(set(idx) - folder_names),
        "imagenes_sin_anotacion_coco": int((df["n_pods"] == 0).sum()),
        "imagenes_en_varias_carpetas": int((df["n_folders"] > 1).sum()),
        "imagenes_con_etiqueta_coco_distinta_a_carpeta": int((~df["label_ok"]).sum()),
        "imagenes_limpias_1_mazorca": int(df["clean_single_pod"].sum()),
        "imagenes_limpias_1_mazorca_por_clase":
            df[df["clean_single_pod"]]["cls"].value_counts().reindex(config.CLASSES).to_dict(),
        "imagenes_limpias_varias_mazorcas_misma_clase":
            int((df["label_ok"] & (df["n_pods"] > 1) & (df["n_folders"] == 1)).sum()),
    }


def repeated_names(df: pd.DataFrame, ds_root: Path) -> pd.DataFrame:
    """Para cada nombre repetido: carpetas, si los archivos son idénticos (MD5) y categorías COCO."""
    rows = []
    for name, g in df[df["file"].duplicated(keep=False)].groupby("file"):
        md5s = {hashlib.md5((ds_root / p).read_bytes()).hexdigest() for p in g["path"]}
        rows.append({"file": name, "carpetas": "|".join(sorted(g["cls"])),
                     "archivos_identicos": len(md5s) == 1,
                     "coco_cats": g["coco_cats"].iloc[0], "n_pods": g["n_pods"].iloc[0]})
    return pd.DataFrame(rows)


# ------------------------------------------------------ 4. particiones oficiales
def official_splits(raw_dir: Path, df: pd.DataFrame) -> tuple[dict, pd.Series]:
    """Lee train/val/test.txt y los cruza con las imágenes por (clase, archivo)."""
    # Clave (carpeta de clase, archivo): hay nombres repetidos en carpetas distintas
    key_to_idx = {(c, n): i for i, (c, n) in df[["cls", "file"]].iterrows()}
    name_to_idx = {n: i for i, n in df["file"].items() if (df["file"] == n).sum() == 1}
    split_col = pd.Series([None] * len(df), index=df.index, dtype=object)
    report = {}
    for split, fname in config.OFFICIAL_SPLIT_FILES.items():
        f = next(iter(raw_dir.rglob(fname)), None)
        if f is None:
            report[split] = "archivo no encontrado"
            continue
        lines = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        matched, unmatched = 0, []
        for line in lines:
            parts = Path(line.replace("\\", "/")).parts
            idx = key_to_idx.get((parts[-2], parts[-1])) if len(parts) >= 2 else None
            if idx is None:
                idx = name_to_idx.get(parts[-1])
            if idx is None:
                # los .txt pueden listar etiquetas YOLO: probar el mismo nombre con extensión de imagen
                stem = Path(line.replace("\\", "/")).stem
                idx = next((i for n, i in name_to_idx.items() if Path(n).stem == stem), None)
            if idx is None:
                unmatched.append(line)
                continue
            if split_col[idx] is not None and split_col[idx] != split:
                split_col[idx] = f"{split_col[idx]}+{split}"  # imagen en dos particiones
            else:
                split_col[idx] = split
            matched += 1
        report[split] = {"lineas": len(lines), "emparejadas": matched,
                         "no_emparejadas": len(unmatched),
                         "ejemplos_no_emparejados": unmatched[:5],
                         "ejemplo_de_linea": lines[0] if lines else None}
    return report, split_col


# ----------------------------------------------------------- 5. casi-duplicados
def near_duplicates(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Pares con distancia de Hamming entre pHash <= threshold (vectorizado con numpy)."""
    h = df["phash"].to_numpy(dtype=np.uint64)
    dist = np.bitwise_count(h[:, None] ^ h[None, :])
    i, j = np.where(np.triu(dist <= max(threshold, 12), k=1))  # candidatos para inspección
    pairs = pd.DataFrame({
        "file_a": df["file"].to_numpy()[i], "cls_a": df["cls"].to_numpy()[i],
        "file_b": df["file"].to_numpy()[j], "cls_b": df["cls"].to_numpy()[j],
        "hamming": dist[i, j],
        "path_a": df["path"].to_numpy()[i], "path_b": df["path"].to_numpy()[j],
    })
    pairs["duplicado"] = pairs["hamming"] <= threshold
    if "official_split" in df:
        pairs["split_a"] = df["official_split"].to_numpy()[i]
        pairs["split_b"] = df["official_split"].to_numpy()[j]
    np.fill_diagonal(dist, 64)
    df["min_hamming"] = dist.min(axis=1)
    return pairs.sort_values("hamming")


# ------------------------------------------------------------------- 6. figuras
def plot_class_distribution(df: pd.DataFrame, out: Path) -> None:
    counts = df["cls"].value_counts().reindex(config.CLASSES)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([f"{c}\n{config.CLASS_NAMES[c].split(' (')[0]}" for c in counts.index],
                  counts.values, color=["#4c956c", "#f4a259", "#bc4b51", "#5b3758"])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v} ({v / counts.sum():.1%})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Imágenes")
    ax.set_title(f"Distribución de clases (n = {counts.sum()})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_sizes(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in config.CLASSES:
        s = df[df["cls"] == c]
        ax.scatter(s["width"], s["height"], s=12, alpha=0.5, label=c)
    ax.set_xlabel("Ancho (px, sin EXIF)")
    ax.set_ylabel("Alto (px, sin EXIF)")
    ax.set_title("Resolución de las imágenes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_samples(df: pd.DataFrame, ds_root: Path, out: Path, n: int = 4) -> None:
    fig, axes = plt.subplots(len(config.CLASSES), n, figsize=(2.6 * n, 2.6 * len(config.CLASSES)))
    for r, c in enumerate(config.CLASSES):
        sample = df[df["cls"] == c].sample(n=min(n, (df["cls"] == c).sum()),
                                           random_state=config.SEED)
        for k, ax in enumerate(axes[r]):
            ax.axis("off")
            if k < len(sample):
                with Image.open(ds_root / sample.iloc[k]["path"]) as im:
                    im.draft("RGB", (400, 400))
                    ax.imshow(ImageOps.exif_transpose(im).convert("RGB"))
            if k == 0:
                ax.set_title(config.CLASS_NAMES[c], fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_hamming(df: pd.DataFrame, threshold: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["min_hamming"], bins=range(0, 34), color="#5b3758")
    ax.axvline(threshold + 0.5, color="red", ls="--", label=f"umbral = {threshold}")
    ax.set_xlabel("Distancia de Hamming al vecino más parecido (pHash, 64 bits)")
    ax.set_ylabel("Imágenes")
    ax.set_title("Detección de casi-duplicados")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_duplicate_pairs(pairs: pd.DataFrame, ds_root: Path, out: Path) -> None:
    """Dos pares por rango de distancia: permite validar visualmente el umbral."""
    bins = [(0, 2), (3, 6), (7, 9), (10, 12)]
    sel = [g.sample(n=min(2, len(g)), random_state=config.SEED)
           for lo, hi in bins if len(g := pairs[pairs["hamming"].between(lo, hi)])]
    if not sel:
        return
    sel = pd.concat(sel)
    fig, axes = plt.subplots(len(sel), 2, figsize=(5, 2.5 * len(sel)))
    axes = np.atleast_2d(axes)
    for r, (_, row) in enumerate(sel.iterrows()):
        for k, side in enumerate("ab"):
            with Image.open(ds_root / row[f"path_{side}"]) as im:
                im.draft("RGB", (300, 300))
                axes[r, k].imshow(ImageOps.exif_transpose(im).convert("RGB"))
            axes[r, k].axis("off")
            axes[r, k].set_title(f"{row[f'cls_{side}']} | d={row['hamming']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------- principal
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=int, default=6,
                    help="distancia de Hamming máxima para considerar casi-duplicados")
    args = ap.parse_args()

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ds_root = find_dataset_root(config.RAW_DIR)
    print(f"Dataset: {ds_root}")

    summary = {"dataset_root": str(ds_root), "inventario": inventory(ds_root),
               "archivos_no_imagen_en_cocoapods_images": non_image_files_in_classes(ds_root)}

    df = image_metadata(ds_root)
    summary["conteo_por_clase"] = df["cls"].value_counts().reindex(config.CLASSES).to_dict()
    summary["total_imagenes"] = len(df)
    summary["desbalance_max_min"] = round(max(summary["conteo_por_clase"].values())
                                          / min(summary["conteo_por_clase"].values()), 3)
    summary["resoluciones_mas_frecuentes"] = (
        (df["width"].astype(str) + "x" + df["height"].astype(str))
        .value_counts().head(5).to_dict())
    summary["orientacion_exif"] = df["exif_orientation"].value_counts().to_dict()
    summary["tamano_archivo_mb"] = df["size_mb"].describe().round(2).to_dict()
    summary["nombres_de_archivo_repetidos"] = int(df["file"].duplicated().sum())

    coco_idx, summary["coco"] = coco_index(ds_root)
    summary["calidad_etiquetas"] = label_quality(df, coco_idx)
    rep = repeated_names(df, ds_root)
    rep.to_csv(config.REPORTS_DIR / "repeated_filenames.csv", index=False)
    summary["nombres_repetidos"] = {
        "nombres": len(rep),
        "archivos_identicos": int(rep["archivos_identicos"].sum()) if len(rep) else 0,
        "combinaciones_de_carpetas": rep["carpetas"].value_counts().to_dict() if len(rep) else {},
    }
    summary["particiones_oficiales"], df["official_split"] = official_splits(config.RAW_DIR, df)
    if df["official_split"].notna().any():
        summary["clases_por_particion_oficial"] = (
            pd.crosstab(df["official_split"].fillna("sin_particion"), df["cls"]).to_dict("index"))

    all_pairs = near_duplicates(df, args.threshold)
    all_pairs.to_csv(config.REPORTS_DIR / "near_duplicates.csv", index=False)
    plot_duplicate_pairs(all_pairs, ds_root, config.FIGURES_DIR / "eda_duplicate_pairs.png")
    # pares del mismo archivo en dos carpetas ya se reportan en nombres_repetidos
    pairs = all_pairs[all_pairs["duplicado"] & (all_pairs["file_a"] != all_pairs["file_b"])]
    summary["casi_duplicados"] = {
        "umbral_hamming": args.threshold,
        "pares_archivos_distintos": len(pairs),
        "pares_misma_clase": int((pairs["cls_a"] == pairs["cls_b"]).sum()),
        "pares_entre_clases": int((pairs["cls_a"] != pairs["cls_b"]).sum()),
        "imagenes_involucradas": int(pd.concat([pairs["file_a"], pairs["file_b"]]).nunique()),
    }
    if "split_a" in pairs:
        cross = pairs["split_a"].notna() & pairs["split_b"].notna() & \
                (pairs["split_a"] != pairs["split_b"])
        summary["casi_duplicados"]["pares_entre_particiones_oficiales"] = int(cross.sum())

    df.drop(columns="phash").to_csv(config.REPORTS_DIR / "images_metadata.csv", index=False)
    (config.REPORTS_DIR / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    plot_class_distribution(df, config.FIGURES_DIR / "eda_class_distribution.png")
    plot_sizes(df, config.FIGURES_DIR / "eda_image_sizes.png")
    plot_samples(df, ds_root, config.FIGURES_DIR / "eda_samples.png")
    plot_hamming(df, args.threshold, config.FIGURES_DIR / "eda_near_duplicates.png")

    print(json.dumps({k: summary[k] for k in
                      ["conteo_por_clase", "total_imagenes", "desbalance_max_min",
                       "orientacion_exif", "nombres_de_archivo_repetidos",
                       "archivos_no_imagen_en_cocoapods_images", "nombres_repetidos",
                       "calidad_etiquetas", "casi_duplicados"]},
                     indent=2, ensure_ascii=False, default=str))
    print(f"\nResumen completo: {config.REPORTS_DIR / 'eda_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())