"""Genera particiones train/val/test sin fuga de información.

Requiere haber ejecutado antes:  python -m scripts.eda

Uso (desde la raíz del repositorio):
    python -m scripts.make_splits                     # umbral de agrupamiento = 12
    python -m scripts.make_splits --group-threshold 10

Pasos:
  1. Filtra imágenes con etiqueta consistente: categoría(s) COCO = carpeta y presentes
     en una sola carpeta de clase (excluye fotos copiadas en dos clases y etiquetas en conflicto).
  2. Agrupa imágenes casi idénticas (pHash, Hamming <= umbral) en componentes conexas:
     todas las fotos de un mismo grupo quedan en la misma partición.
  3. StratifiedGroupKFold: ~71 % train / ~14 % val / ~14 % test, estratificado por clase.

Salidas:
    data/splits/{train,val,test}.csv    path (relativo al dataset), cls, label, group
    data/splits/excluded.csv            imágenes excluidas y motivo
    reports/splits_summary.json         conteos, verificación de fugas y comparación con las oficiales
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from cacao import config


def connected_groups(paths: list[str], pairs: pd.DataFrame) -> dict[str, int]:
    """Union-find: cada componente conexa del grafo de casi-duplicados es un grupo."""
    parent = {p: p for p in paths}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs["path_a"], pairs["path_b"]):
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    roots = {find(p) for p in paths}
    root_id = {r: i for i, r in enumerate(sorted(roots))}
    return {p: root_id[find(p)] for p in paths}


def cross_split_pairs(pairs: pd.DataFrame, split_of: dict[str, str]) -> int:
    sa = pairs["path_a"].map(split_of)
    sb = pairs["path_b"].map(split_of)
    return int((sa.notna() & sb.notna() & (sa != sb)).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group-threshold", type=int, default=12,
                    help="distancia de Hamming máxima para agrupar imágenes (máx. 12: "
                         "es el límite de candidatos que guarda el EDA)")
    args = ap.parse_args()

    meta_f = config.REPORTS_DIR / "images_metadata.csv"
    pairs_f = config.REPORTS_DIR / "near_duplicates.csv"
    if not meta_f.exists() or not pairs_f.exists():
        print("ERROR: falta la salida del EDA. Ejecuten: python -m scripts.eda")
        return 1
    meta = pd.read_csv(meta_f)
    pairs = pd.read_csv(pairs_f)
    pairs = pairs[pairs["hamming"] <= args.group_threshold]

    # 1. Filtro de etiquetas -------------------------------------------------------
    meta["motivo_exclusion"] = np.select(
        [meta["n_folders"] > 1, ~meta["label_ok"]],
        ["misma_foto_en_varias_carpetas", "categoria_coco_distinta_a_carpeta"],
        default="")
    excluded = meta[meta["motivo_exclusion"] != ""]
    df = meta[meta["motivo_exclusion"] == ""].reset_index(drop=True)

    # 2. Grupos de casi-duplicados -------------------------------------------------
    group_of = connected_groups(df["path"].tolist(), pairs)
    df["group"] = df["path"].map(group_of)
    df["label"] = df["cls"].map({c: i for i, c in enumerate(config.CLASSES)})

    # 3. Particiones estratificadas por grupo ---------------------------------------
    sgkf_test = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=config.SEED)
    rest_idx, test_idx = next(sgkf_test.split(df, df["label"], df["group"]))
    rest = df.iloc[rest_idx]
    sgkf_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=config.SEED)
    tr_rel, val_rel = next(sgkf_val.split(rest, rest["label"], rest["group"]))
    df["split"] = ""
    df.loc[rest.index[tr_rel], "split"] = "train"
    df.loc[rest.index[val_rel], "split"] = "val"
    df.loc[df.index[test_idx], "split"] = "test"

    # Verificaciones ---------------------------------------------------------------
    groups_per_split = df.groupby("group")["split"].nunique()
    assert (groups_per_split == 1).all(), "Un grupo quedó repartido en varias particiones"
    split_of = dict(zip(df["path"], df["split"]))
    our_leaks = cross_split_pairs(pairs, split_of)
    assert our_leaks == 0, f"{our_leaks} pares de casi-duplicados entre particiones"

    official = meta.dropna(subset=["official_split"])
    official_leaks = cross_split_pairs(pairs, dict(zip(official["path"],
                                                       official["official_split"])))
    dup_files = meta[meta["n_folders"] > 1]
    official_same_photo_cross = int(
        dup_files.groupby("file")["official_split"].nunique().gt(1).sum())

    # Salidas ----------------------------------------------------------------------
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["path", "cls", "label", "group"]
    for s in ["train", "val", "test"]:
        df[df["split"] == s][cols].to_csv(config.SPLITS_DIR / f"{s}.csv", index=False)
    excluded[["path", "cls", "coco_cats", "n_pods", "motivo_exclusion"]].to_csv(
        config.SPLITS_DIR / "excluded.csv", index=False)

    counts = pd.crosstab(df["split"], df["cls"]).reindex(["train", "val", "test"])
    summary = {
        "umbral_agrupamiento_hamming": args.group_threshold,
        "imagenes_totales": len(meta),
        "imagenes_excluidas": excluded["motivo_exclusion"].value_counts().to_dict(),
        "imagenes_usadas": len(df),
        "imagenes_usadas_por_clase": df["cls"].value_counts().reindex(config.CLASSES).to_dict(),
        "grupos": int(df["group"].nunique()),
        "grupos_con_2+_imagenes": int((df["group"].value_counts() > 1).sum()),
        "imagenes_en_grupos_2+": int(df["group"].map(df["group"].value_counts()).gt(1).sum()),
        "conteo_por_particion_y_clase": counts.to_dict("index"),
        "proporcion_por_particion": (counts.sum(axis=1) / len(df)).round(3).to_dict(),
        "verificacion": {
            "pares_casi_duplicados_entre_particiones_propias": our_leaks,
            "pares_casi_duplicados_entre_particiones_oficiales": official_leaks,
            "misma_foto_en_dos_carpetas_y_en_dos_particiones_oficiales":
                official_same_photo_cross,
        },
    }
    (config.REPORTS_DIR / "splits_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
