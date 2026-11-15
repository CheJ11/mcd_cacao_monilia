"""Grad-CAM (Selvaraju et al., 2017) para los modelos de cacao.model.

Responde: ¿en qué región de la imagen se basó el modelo para su predicción?
Se usa en evaluate.py (análisis de errores) y en la app (explicación al usuario).
"""
import keras
import matplotlib
import numpy as np
import tensorflow as tf

from cacao.model import BACKBONES

# Última capa convolucional de cada backbone (su salida espacial se pondera con los gradientes)
LAST_CONV = {"mobilenetv2": "out_relu", "vgg16": "block5_conv3"}


def _backbone(model: keras.Model) -> tuple[str, keras.Model]:
    for layer in model.layers:
        for name in BACKBONES:
            if layer.name.startswith(name):
                return name, layer
    raise ValueError("No se encontró el backbone dentro del modelo")


def grad_cam(model: keras.Model, images: np.ndarray, class_idx=None) -> tuple[np.ndarray, np.ndarray]:
    """images: (n, 512, 512, 3) en 0-255. Devuelve (mapas (n, h, w) en [0, 1], probabilidades).

    Recorre el modelo capa a capa sin el bloque de aumento de datos (inactivo en inferencia)
    para obtener la activación de la última capa convolucional y su gradiente.
    """
    name, base = _backbone(model)
    last = base.get_layer(LAST_CONV[name])
    conv_model = keras.Model(base.inputs, last.output)
    tail = base.layers[base.layers.index(last) + 1:]  # capas posteriores (p. ej. block5_pool)
    resize, norm = model.get_layer("redimensionar"), model.get_layer("normalizar")
    pool, dense = model.get_layer("pooling"), model.get_layer("clases")

    x = tf.convert_to_tensor(images, dtype=tf.float32)
    conv = conv_model(norm(resize(x)), training=False)
    with tf.GradientTape() as tape:
        # Se observa la activación ANTES de usarla: el gradiente fluye aunque las
        # capas del backbone estén congeladas (variables no entrenables no se rastrean)
        tape.watch(conv)
        h = conv
        for layer in tail:
            h = layer(h, training=False)
        # Puntaje ANTES del softmax (logit), como en el artículo original: con predicciones
        # muy confiadas la probabilidad se satura en 1 y su gradiente se anula
        logits = tf.matmul(pool(h), dense.kernel) + dense.bias
        probs = tf.nn.softmax(logits)
        idx = tf.argmax(probs, axis=1) if class_idx is None else \
            tf.fill([tf.shape(probs)[0]], tf.cast(class_idx, tf.int64))
        score = tf.gather(logits, idx, axis=1, batch_dims=1)
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)      # importancia por canal
    cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1)).numpy()  # (n, h, w)
    cam /= cam.max(axis=(1, 2), keepdims=True) + 1e-8
    return cam, probs.numpy()


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Superpone el mapa (redimensionado al tamaño de la imagen) con la paleta 'jet'."""
    h, w = image.shape[:2]
    cam_big = tf.image.resize(cam[..., None], (h, w), method="bilinear").numpy()[..., 0]
    heat = matplotlib.colormaps["jet"](cam_big)[..., :3] * 255
    return np.clip((1 - alpha) * image + alpha * heat, 0, 255).astype(np.uint8)
