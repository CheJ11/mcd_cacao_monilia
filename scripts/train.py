"""Entrena un clasificador por transfer learning en dos fases.

Requiere: python -m scripts.preprocess

Uso (desde la raíz del repositorio):
    python -m scripts.train --model mobilenetv2
    python -m scripts.train --model vgg16
    python -m scripts.train --model mobilenetv2 --limit 64 --epochs-head 1 --epochs-ft 1   # prueba rápida
    python -m scripts.train --model mobilenetv2 --img-size 384 --tag 384   # experimento: guarda mobilenetv2_384

Fases:
  1. Backbone congelado: solo se entrena el clasificador (lr alto).
  2. Fine-tuning: se descongelan las últimas capas del backbone (lr 100 veces menor).
  Se conserva el mejor modelo según la pérdida en validación de ambas fases.
  El conjunto de TEST no se usa aquí (se evalúa una sola vez en evaluate.py).

Salidas:
    (<nombre> = <modelo> o <modelo>_<tag> si se usa --tag)
    models/<nombre>.keras                       modelo final sin estado del optimizador
    reports/history_<nombre>.csv                pérdida y exactitud por época y fase
    reports/train_<nombre>.json                 configuración, tiempos y métricas de validación
    reports/figures/train_<nombre>_curves.png   curvas de aprendizaje
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
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from cacao import config
from cacao.model import BACKBONES, build_model, unfreeze_top

SIZE = config.PREPROCESS_SIZE


def make_dataset(split: str, batch: int, shuffle: bool, limit: int | None):
    df = pd.read_csv(config.SPLITS_DIR / f"{split}.csv")
    if limit:
        df = df.groupby("cls", group_keys=False).head(max(1, limit // len(config.CLASSES)))
    root = config.PROCESSED_DIR / str(SIZE)
    paths = [str(root / p) for p in df["path"]]

    def load(path, label):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        return tf.ensure_shape(tf.cast(img, tf.float32), [SIZE, SIZE, 3]), label

    ds = tf.data.Dataset.from_tensor_slices((paths, df["label"].to_numpy()))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE), df


def fit_phase(model, train_ds, val_ds, epochs, lr, class_weight, ckpt, patience, threshold=None):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True,
                                        save_weights_only=True,
                                        initial_value_threshold=threshold),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience),
    ]
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     class_weight=class_weight, callbacks=callbacks, verbose=2)
    return pd.DataFrame(hist.history), time.time() - t0


def plot_curves(hist: pd.DataFrame, name: str, out) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m in zip(axes, ["loss", "accuracy"]):
        ax.plot(hist["epoca"], hist[m], marker="o", ms=3, label="train")
        ax.plot(hist["epoca"], hist[f"val_{m}"], marker="o", ms=3, label="val")
        ft = hist.loc[hist["fase"] == "fine_tuning", "epoca"]
        if len(ft):
            ax.axvline(ft.min() - 0.5, color="gray", ls="--", label="inicio fine-tuning")
        ax.set_xlabel("Época")
        ax.set_title({"loss": "Pérdida (entropía cruzada)", "accuracy": "Exactitud"}[m])
        ax.legend()
    fig.suptitle(f"Curvas de aprendizaje: {name}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(BACKBONES))
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE[0])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs-head", type=int, default=20)
    ap.add_argument("--epochs-ft", type=int, default=15)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--fine-tune-layers", type=int, default=None)
    ap.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    ap.add_argument("--tag", default=None,
                    help="sufijo para no sobrescribir otros experimentos (p. ej. 384)")
    ap.add_argument("--limit", type=int, default=None,
                    help="usar solo ~N imágenes por partición (prueba rápida)")
    args = ap.parse_args()

    if not (config.PROCESSED_DIR / str(SIZE) / "manifest.json").exists():
        print("ERROR: faltan las imágenes preprocesadas. Ejecuten: python -m scripts.preprocess")
        return 1
    keras.utils.set_random_seed(config.SEED)
    name = args.model                                       # backbone
    out_name = f"{name}_{args.tag}" if args.tag else name    # nombre de los archivos
    tmp = config.MODELS_DIR / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, train_df = make_dataset("train", args.batch, True, args.limit)
    val_ds, val_df = make_dataset("val", args.batch, False, args.limit)
    cw = compute_class_weight("balanced", classes=np.arange(len(config.CLASSES)),
                              y=train_df["label"])
    class_weight = dict(enumerate(cw))
    print("Pesos por clase:", {c: round(w, 3) for c, w in zip(config.CLASSES, cw)})

    weights = None if args.weights == "none" else "imagenet"
    model, base = build_model(name, args.img_size, weights=weights)

    # Fase 1: backbone congelado ----------------------------------------------------
    ckpt1 = tmp / f"{out_name}_fase1.weights.h5"
    h1, t1 = fit_phase(model, train_ds, val_ds, args.epochs_head, args.lr_head,
                       class_weight, ckpt1, args.patience)
    model.load_weights(ckpt1)
    best1 = float(h1["val_loss"].min())

    # Fase 2: fine-tuning ------------------------------------------------------------
    n_ft = args.fine_tune_layers or BACKBONES[name]["fine_tune_layers"]
    n_params_ft = unfreeze_top(base, n_ft)
    ckpt2 = tmp / f"{out_name}_fase2.weights.h5"
    ckpt2.unlink(missing_ok=True)
    h2, t2 = fit_phase(model, train_ds, val_ds, args.epochs_ft, args.lr_ft,
                       class_weight, ckpt2, args.patience, threshold=best1)
    improved = ckpt2.exists()  # solo se guarda si mejora la mejor pérdida de la fase 1
    best_ckpt = ckpt2 if improved else ckpt1

    # Modelo final: misma arquitectura, sin optimizador (archivo más liviano) --------
    final, _ = build_model(name, args.img_size, weights=None)
    final.load_weights(best_ckpt)
    out_path = config.MODELS_DIR / f"{out_name}.keras"
    final.save(out_path)

    # Verificación: el archivo guardado reproduce las predicciones ------------------
    reloaded = keras.models.load_model(out_path)
    p_mem = final.predict(val_ds, verbose=0)
    p_disk = reloaded.predict(val_ds, verbose=0)
    assert np.allclose(p_mem, p_disk, atol=1e-5), "El modelo guardado no reproduce las predicciones"
    y_val, y_hat = val_df["label"].to_numpy(), p_disk.argmax(axis=1)

    hist = pd.concat([h1.assign(fase="cabeza"), h2.assign(fase="fine_tuning")],
                     ignore_index=True)
    hist.insert(0, "epoca", np.arange(1, len(hist) + 1))
    hist.to_csv(config.REPORTS_DIR / f"history_{out_name}.csv", index=False)
    plot_curves(hist, out_name, config.FIGURES_DIR / f"train_{out_name}_curves.png")

    summary = {
        "modelo": out_name, "backbone": name, "img_size": args.img_size, "batch": args.batch,
        "lr_cabeza": args.lr_head, "lr_fine_tuning": args.lr_ft,
        "capas_descongeladas": n_ft, "parametros_entrenables_fine_tuning": n_params_ft,
        "parametros_totales": int(final.count_params()),
        "epocas_fase1": len(h1), "epocas_fase2": len(h2),
        "segundos_fase1": round(t1, 1), "segundos_fase2": round(t2, 1),
        "fine_tuning_mejoro_val_loss": improved,
        "pesos_por_clase": {c: round(w, 3) for c, w in zip(config.CLASSES, cw)},
        "val_loss_min": round(float(hist["val_loss"].min()), 4),
        "val_accuracy": round(accuracy_score(y_val, y_hat), 4),
        "val_f1_macro": round(f1_score(y_val, y_hat, average="macro"), 4),
        "tamano_modelo_mb": round(out_path.stat().st_size / 1e6, 1),
        "limit": args.limit, "weights": args.weights,
    }
    (config.REPORTS_DIR / f"train_{out_name}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
