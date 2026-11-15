"""Evalúa uno o más modelos entrenados sobre una partición.

Protocolo (evita sesgo de selección):
  1. Comparar modelos en VALIDACIÓN y elegir uno.
  2. Evaluar SOLO el modelo elegido en TEST, una sola vez.

Uso (desde la raíz del repositorio):
    python -m scripts.evaluate --models mobilenetv2 vgg16          # validación (por defecto)
    python -m scripts.evaluate --models mobilenetv2 mobilenetv2_384  # experimentos con --tag
    python -m scripts.evaluate --models mobilenetv2 --split test   # test del modelo elegido

Salidas por modelo y partición:
    reports/eval_<modelo>_<split>.json             métricas, IC 95 % bootstrap, por clase, confianza
    reports/predictions_<modelo>_<split>.csv       una fila por imagen con probabilidades
    reports/figures/eval_<modelo>_<split>_confusion.png
    reports/figures/eval_<modelo>_<split>_gradcam.png
Con dos modelos: reports/compare_<a>_vs_<b>_<split>.json (diferencia pareada con IC 95 %).
"""
import argparse
import json
import sys
import time

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

import cacao.model  # noqa: F401  (registra la capa ImagenetPreprocess para load_model)
from cacao import config
from cacao.explain import grad_cam, overlay
from scripts.train import make_dataset

N_BOOT = 2000


def bootstrap_ci(y, pred, metric, rng, n=N_BOOT):
    idx = rng.integers(0, len(y), size=(n, len(y)))
    vals = np.array([metric(y[i], pred[i]) for i in idx])
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def f1_macro(y, p):
    return f1_score(y, p, average="macro", labels=range(len(config.CLASSES)), zero_division=0)


def plot_confusion(cm: np.ndarray, title: str, out) -> None:
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    fontsize=9, color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(config.CLASSES)), config.CLASSES)
    ax.set_yticks(range(len(config.CLASSES)), config.CLASSES)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Proporción por fila (recall)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_gradcam(model, pred_df: pd.DataFrame, title: str, out, per_class: int = 3) -> None:
    """Por clase real: hasta 2 aciertos y 1 error (si existe), con su mapa Grad-CAM."""
    rng = np.random.default_rng(config.SEED)
    rows = []
    for c in config.CLASSES:
        sub = pred_df[pred_df["cls"] == c]
        ok, bad = sub[sub["correcto"]], sub[~sub["correcto"]]
        pick = [ok.iloc[i] for i in rng.permutation(len(ok))[:per_class - min(1, len(bad))]]
        pick += [bad.iloc[i] for i in rng.permutation(len(bad))[:1]]
        rows += pick
    root = config.PROCESSED_DIR / str(config.PREPROCESS_SIZE)
    imgs = np.stack([np.asarray(Image.open(root / r["path"]).convert("RGB"), dtype=np.float32)
                     for r in rows])
    cams, _ = grad_cam(model, imgs)  # mapa de la clase predicha
    n_cols = per_class
    fig, axes = plt.subplots(len(config.CLASSES), n_cols, figsize=(3 * n_cols, 3.2 * len(config.CLASSES)))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    k = 0
    for r_i, c in enumerate(config.CLASSES):
        for col in range(n_cols):
            if k < len(rows) and rows[k]["cls"] == c:
                r = rows[k]
                axes[r_i, col].imshow(overlay(imgs[k], cams[k]))
                color = "green" if r["correcto"] else "red"
                axes[r_i, col].set_title(f"real {r['cls']} → pred {r['pred']} ({r['confianza']:.0%})",
                                         fontsize=9, color=color)
                k += 1
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def evaluate(name: str, split: str, rng) -> tuple[dict, np.ndarray, np.ndarray]:
    path = config.MODELS_DIR / f"{name}.keras"
    model = keras.models.load_model(path)
    ds, df = make_dataset(split, 32, False, None)

    model.predict(ds.take(1), verbose=0)  # calentamiento (compilación del grafo)
    t0 = time.perf_counter()
    probs = model.predict(ds, verbose=0)
    ms_img = (time.perf_counter() - t0) / len(df) * 1000

    y, pred = df["label"].to_numpy(), probs.argmax(axis=1)
    conf = probs.max(axis=1)
    cm = confusion_matrix(y, pred, labels=range(len(config.CLASSES)))
    report = classification_report(y, pred, labels=range(len(config.CLASSES)),
                                   target_names=config.CLASSES, output_dict=True, zero_division=0)

    # Confianza: si la app solo responde cuando la confianza supera un umbral,
    # ¿qué fracción de fotos cubre y con qué exactitud?
    confidence = {}
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        m = conf >= t
        confidence[str(t)] = {"cobertura": round(float(m.mean()), 3),
                              "exactitud": round(float((pred[m] == y[m]).mean()), 3) if m.any() else None}

    pred_df = df[["path", "cls"]].copy()
    pred_df["pred"] = [config.CLASSES[i] for i in pred]
    pred_df["confianza"] = conf.round(4)
    pred_df["correcto"] = pred == y
    for i, c in enumerate(config.CLASSES):
        pred_df[f"p_{c}"] = probs[:, i].round(4)
    pred_df.to_csv(config.REPORTS_DIR / f"predictions_{name}_{split}.csv", index=False)

    tag = f"{name} | {split} (n = {len(df)})"
    plot_confusion(cm, tag, config.FIGURES_DIR / f"eval_{name}_{split}_confusion.png")
    plot_gradcam(model, pred_df, f"Grad-CAM: {tag}",
                 config.FIGURES_DIR / f"eval_{name}_{split}_gradcam.png")

    result = {
        "modelo": name, "particion": split, "n": len(df),
        "accuracy": round(accuracy_score(y, pred), 4),
        "accuracy_ic95": bootstrap_ci(y, pred, accuracy_score, rng),
        "f1_macro": round(f1_macro(y, pred), 4),
        "f1_macro_ic95": bootstrap_ci(y, pred, f1_macro, rng),
        "balanced_accuracy": round(balanced_accuracy_score(y, pred), 4),
        "por_clase": {c: {k: round(v, 4) for k, v in report[c].items()} for c in config.CLASSES},
        "matriz_confusion": cm.tolist(),
        "confusiones_mas_frecuentes": sorted(
            [{"real": config.CLASSES[i], "pred": config.CLASSES[j], "n": int(cm[i, j])}
             for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0],
            key=lambda d: -d["n"])[:5],
        "analisis_confianza": confidence,
        "ms_por_imagen_cpu": round(ms_img, 2),
        "tamano_modelo_mb": round(path.stat().st_size / 1e6, 1),
        "parametros": int(model.count_params()),
    }
    (config.REPORTS_DIR / f"eval_{name}_{split}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result, y, pred


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="nombres de archivo en models/ sin .keras (p. ej. mobilenetv2 mobilenetv2_384)")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    args = ap.parse_args()
    if args.split == "test" and len(args.models) > 1:
        print("ERROR: en test se evalúa solo el modelo ya elegido en validación.")
        return 1
    missing = [m for m in args.models if not (config.MODELS_DIR / f"{m}.keras").exists()]
    if missing:
        print(f"ERROR: faltan modelos {missing}. Ejecuten: python -m scripts.train --model ...")
        return 1
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    outs = {}
    for name in args.models:
        res, y, pred = evaluate(name, args.split, np.random.default_rng(config.SEED))
        outs[name] = (y, pred)
        short = {k: res[k] for k in ["accuracy", "accuracy_ic95", "f1_macro", "f1_macro_ic95",
                                     "balanced_accuracy", "confusiones_mas_frecuentes",
                                     "ms_por_imagen_cpu", "tamano_modelo_mb"]}
        short["recall_por_clase"] = {c: res["por_clase"][c]["recall"] for c in config.CLASSES}
        print(f"\n=== {name} | {args.split} ===")
        print(json.dumps(short, indent=2, ensure_ascii=False))

    if len(args.models) == 2:
        (a, (y, pa)), (b, (_, pb)) = outs.items()
        rng = np.random.default_rng(config.SEED)
        idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
        diff = np.array([f1_macro(y[i], pa[i]) - f1_macro(y[i], pb[i]) for i in idx])
        comp = {
            "particion": args.split,
            "diferencia_f1_macro": f"{a} - {b}",
            "valor": round(float(f1_macro(y, pa) - f1_macro(y, pb)), 4),
            "ic95_bootstrap_pareado": [round(float(np.percentile(diff, 2.5)), 4),
                                       round(float(np.percentile(diff, 97.5)), 4)],
            "desacuerdos_entre_modelos": int((pa != pb).sum()),
        }
        comp["interpretacion"] = ("el IC incluye 0: diferencia no concluyente con este n"
                                  if comp["ic95_bootstrap_pareado"][0] <= 0 <= comp["ic95_bootstrap_pareado"][1]
                                  else "el IC excluye 0: diferencia consistente")
        (config.REPORTS_DIR / f"compare_{a}_vs_{b}_{args.split}.json").write_text(
            json.dumps(comp, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n=== Comparación pareada ===")
        print(json.dumps(comp, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
