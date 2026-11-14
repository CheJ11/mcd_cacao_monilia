"""Modelos de transfer learning (Actividad 8) con el preprocesamiento DENTRO del modelo.

Entrada del modelo: imagen RGB cuadrada de config.PREPROCESS_SIZE px, valores 0-255
(exactamente lo que produce cacao.data.load_square_image). El modelo se encarga de:
    aumento de datos (solo en entrenamiento) -> redimensionado -> normalización ImageNet
    -> backbone preentrenado -> clasificador.
Así la app de Streamlit no puede aplicar una normalización distinta a la del entrenamiento.
"""
import keras
from keras import layers

from cacao import config

# Normalización que espera cada backbone (la misma que su preprocess_input de Keras)
#   "tf":    x / 127.5 - 1                      (MobileNetV2)
#   "caffe": RGB -> BGR y resta de la media ImageNet por canal (VGG16)
BACKBONES = {
    "mobilenetv2": {"fn": keras.applications.MobileNetV2, "mode": "tf",
                    "fine_tune_layers": 30},     # últimos bloques "inverted residual"
    "vgg16": {"fn": keras.applications.VGG16, "mode": "caffe",
              "fine_tune_layers": 4},            # block5: 3 convoluciones + pooling
}


@keras.saving.register_keras_serializable(package="cacao")
class ImagenetPreprocess(layers.Layer):
    """Equivalente serializable de keras.applications.<modelo>.preprocess_input."""

    def __init__(self, mode: str, **kwargs):
        super().__init__(**kwargs)
        if mode not in ("tf", "caffe"):
            raise ValueError(f"mode desconocido: {mode}")
        self.mode = mode

    def call(self, x):
        if self.mode == "tf":
            return x / 127.5 - 1.0
        bgr = keras.ops.flip(x, axis=-1)
        return bgr - keras.ops.convert_to_tensor([103.939, 116.779, 123.68], dtype=x.dtype)

    def get_config(self):
        return {**super().get_config(), "mode": self.mode}


def augmentation() -> keras.Sequential:
    """Variaciones plausibles en campo. Sin cambios de tono/saturación: el color
    (manchas marrones, esporulación blanca) es justamente la señal diagnóstica."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),          # ±18°
        layers.RandomZoom((-0.15, 0.15)),
        layers.RandomBrightness(0.15),        # rango de valores 0-255 (por defecto)
        layers.RandomContrast(0.15),
    ], name="aumento")


def build_model(name: str, img_size: int = config.IMG_SIZE[0], dropout: float = 0.3,
                weights: str | None = "imagenet") -> tuple[keras.Model, keras.Model]:
    """Devuelve (modelo completo, backbone). El backbone empieza congelado."""
    spec = BACKBONES[name]
    inputs = keras.Input((config.PREPROCESS_SIZE, config.PREPROCESS_SIZE, 3), name="imagen")
    x = layers.Resizing(img_size, img_size, interpolation="bilinear", antialias=True,
                        name="redimensionar")(inputs)
    x = augmentation()(x)
    x = ImagenetPreprocess(spec["mode"], name="normalizar")(x)
    base = spec["fn"](include_top=False, weights=weights, input_shape=(img_size, img_size, 3))
    base.trainable = False
    # training=False: BatchNorm del backbone usa sus estadísticas de ImageNet también
    # durante el fine-tuning (recomendación de la guía de transfer learning de Keras)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="pooling")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(len(config.CLASSES), activation="softmax", name="clases")(x)
    return keras.Model(inputs, outputs, name=f"cacao_{name}"), base


def unfreeze_top(base: keras.Model, n_layers: int) -> int:
    """Descongela las últimas n capas del backbone, excepto BatchNormalization."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return sum(int(keras.ops.size(w)) for w in base.trainable_weights)
