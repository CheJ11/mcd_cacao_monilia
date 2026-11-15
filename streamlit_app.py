"""Demo: detección del estadio de moniliasis en mazorcas de cacao.

Ejecución local (desde la raíz del repositorio):
    streamlit run streamlit_app.py
"""
import json
import time

import numpy as np
import pandas as pd
import streamlit as st

from cacao import config
from cacao.data import load_square_image

AUTORES = "Completar con los nombres de los autores"
REPO_URL = "https://github.com/CheJ11/mcd_cacao_monilia"
TH = config.CONFIDENCE_THRESHOLD

ORIENTACION = {
    "h0": "No se observan síntomas. Mantenga la inspección periódica: los síntomas "
          "iniciales de la moniliasis son difíciles de distinguir a simple vista.",
    "m1": "Posible infección temprana (deformaciones o jorobas). La mazorca puede estar "
          "infectada aunque aún no tenga manchas: márquela y revísela en la próxima ronda "
          "de remoción de mazorcas enfermas.",
    "m2": "Infección establecida (manchas). Retire la mazorca antes de que aparezca el "
          "polvo blanco de esporas.",
    "m3": "La mazorca está esporulando y dispersa la enfermedad. Retírela con cuidado para "
          "no esparcir las esporas y no la deje expuesta en la plantación.",
}
NIVEL = {"h0": st.success, "m1": st.warning, "m2": st.error, "m3": st.error}

st.set_page_config(page_title="Monilia en cacao", page_icon="🍫", layout="wide")


# ----------------------------------------------------------------- recursos (caché)
@st.cache_resource(show_spinner="Cargando el modelo…")
def load_model():
    import keras
    import cacao.model  # noqa: F401  registra la capa de normalización para load_model
    return keras.models.load_model(config.MODELS_DIR / f"{config.APP_MODEL}.keras")


@st.cache_data
def read_json(name: str):
    f = config.REPORTS_DIR / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


@st.cache_data
def read_gallery() -> list[dict]:
    f = config.GALLERY_DIR / "gallery.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def figure(name: str, caption: str | None = None) -> None:
    f = config.FIGURES_DIR / name
    if f.exists():
        st.image(str(f), caption=caption, width="stretch")
    else:
        st.info(f"Figura no disponible: {name}")


def pct(x: float) -> str:
    return f"{x:.0%}"


# ------------------------------------------------------- utilidades de navegador (JS)
def run_js(code: str) -> None:
    """Ejecuta JavaScript en el navegador del usuario. El sello de tiempo hace que el
    elemento sea distinto en cada ejecución, de modo que el script vuelva a correr."""
    st.html(f"<script>/* {time.time_ns()} */ {code}</script>", unsafe_allow_javascript=True)


def anchor(anchor_id: str) -> None:
    """Marca invisible a la que se puede desplazar la página."""
    st.html(f'<div id="{anchor_id}"></div>')


def scroll_to(anchor_id: str) -> None:
    """Desplaza la página hasta la marca. Reintenta mientras la marca no exista y vuelve
    a verificar después: el resto del contenido puede seguir dibujándose y moverla."""
    run_js(f"""
      (function () {{
        let tries = 0, checks = 0;
        const go = () => {{
          const el = document.getElementById("{anchor_id}");
          if (!el) {{ if (tries++ < 50) setTimeout(go, 100); return; }}
          el.scrollIntoView({{behavior: "smooth", block: "start"}});
          if (checks++ < 6) setTimeout(() => {{
            if (Math.abs(el.getBoundingClientRect().top) > 40) go();
          }}, 700);
        }};
        setTimeout(go, 150);
      }})();""")


def theme_button() -> None:
    """Alterna modo claro/oscuro. Escribe la misma preferencia que guarda el menú
    Settings de Streamlit (localStorage, formato verificado en Streamlit 1.67) y recarga.
    El tema actual se lee del color de fondo real de la página."""
    if st.button("🌓 Claro / oscuro", key="tema", width="stretch",
                 help="Cambia entre modo claro y oscuro (recarga la página)"):
        run_js("""
          const app = document.querySelector('[data-testid="stApp"]');
          const rgb = getComputedStyle(app).backgroundColor.match(/[0-9]+/g).map(Number);
          const isDark = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) < 128;
          const key = `stActiveTheme-${window.location.pathname}-v2`;
          window.localStorage.setItem(key, JSON.stringify(isDark ? "Light" : "Dark"));
          window.location.reload();""")


def pick_example(i: int) -> None:
    st.session_state["galeria"] = i
    st.session_state["scroll"] = "resultado"


def back_to_gallery() -> None:
    st.session_state["scroll"] = "galeria"


# --------------------------------------------------------------------- diagnóstico
def analyze(img, real_cls: str | None = None) -> None:
    """Muestra predicción, probabilidades, orientación y Grad-CAM para una imagen 512x512."""
    model = load_model()
    x = np.asarray(img, dtype=np.float32)[None]
    with st.spinner("Analizando la imagen…"):
        probs = model(x, training=False).numpy()[0]
    k = int(probs.argmax())
    cls, conf, p_inf = config.CLASSES[k], float(probs[k]), float(1 - probs[0])

    col_img, col_res = st.columns([1, 1], gap="large")
    with col_img:
        show_cam = st.toggle("Mostrar dónde miró el modelo (Grad-CAM)", value=False)
        if show_cam:
            from cacao.explain import grad_cam, overlay
            with st.spinner("Calculando el mapa de atención…"):
                cam, _ = grad_cam(model, x)
            st.image(overlay(np.asarray(img), cam[0]), width="stretch",
                     caption="Rojo: regiones que más influyeron en la predicción")
        else:
            st.image(img, width="stretch", caption="Imagen tal como la recibe el modelo")

    with col_res:
        if conf >= TH:
            NIVEL[cls](f"**{config.CLASS_NAMES[cls]}** ({cls})")
        else:
            st.warning("**Resultado incierto.** La confianza es menor al "
                       f"{pct(TH)}. Tome otra foto: una sola mazorca, de cerca y con buena luz.")
        m1, m2 = st.columns(2)
        m1.metric("Confianza", pct(conf))
        m2.metric("Probabilidad de infección", pct(p_inf),
                  help="1 − p(sana) = p(m1) + p(m2) + p(m3)")
        for c, p in zip(config.CLASSES, probs):
            st.progress(float(p), text=f"{c} · {config.CLASS_NAMES[c]}: {p:.1%}")

        if conf >= TH:
            st.info(f"**Orientación general:** {ORIENTACION[cls]}")
        if cls == "h0":
            test = read_json(f"eval_{config.APP_MODEL}_test.json")
            if test:
                cm = np.array(test["matriz_confusion"])
                st.caption(f"Precaución: el error más frecuente del modelo es clasificar "
                           f"infecciones tempranas como sanas ({cm[1, 0] / cm[1].sum():.0%} "
                           "de las m1 en el conjunto de test). Revise la probabilidad de infección.")
        if real_cls is not None:
            real = f"Etiqueta real: **{real_cls}** ({config.CLASS_NAMES[real_cls]}). "
            if conf < TH:
                st.info(real + f"La clase más probable era {cls}, con confianza baja: "
                               "la app no emite un veredicto.")
            elif real_cls == cls:
                st.success(real + "El modelo acertó.")
            else:
                st.error(real + "El modelo se equivocó en esta imagen.")
        st.caption("Herramienta de apoyo, no reemplaza el diagnóstico de un técnico agrícola. "
                   "Valide la orientación con un técnico o el INIAP.")


def tab_diagnostico() -> None:
    with st.expander("¿Cómo usar la app?", icon="ℹ️"):
        st.markdown(
            "1. Elija una imagen: un ejemplo de la galería, una foto propia o la cámara.\n"
            "2. Para fotos propias: **una sola mazorca de cacao**, de cerca y con buena luz.\n"
            "3. Lea el estadio, la confianza y la probabilidad de infección. Active Grad-CAM "
            "para ver en qué zona se fijó el modelo.\n\n"
            "El modelo solo conoce mazorcas de cacao: con otro tipo de imagen igual responderá "
            "con alguno de los 4 estadios, pero esa respuesta no tiene sentido.")
    source = st.radio("Origen de la imagen",
                      ["Galería de ejemplos", "Subir una foto", "Tomar una foto"],
                      horizontal=True)

    if source == "Galería de ejemplos":
        gallery = read_gallery()
        if not gallery:
            st.info("Galería no disponible. Ejecuten: python -m scripts.make_gallery")
            return
        anchor("galeria")
        st.caption("Imágenes del conjunto de **test**: el modelo nunca las vio al entrenar. "
                   "¿Puede adivinar el estadio antes de analizarla?")
        sel = st.session_state.get("galeria")
        cols = st.columns(4)
        for i, item in enumerate(gallery):
            with cols[i % 4]:
                st.image(str(config.GALLERY_DIR / item["archivo"]), width="stretch")
                st.button(f"Analizar ejemplo {i + 1}", key=f"g{i}", width="stretch",
                          type="primary" if i == sel else "secondary",
                          on_click=pick_example, args=(i,))
        if sel is not None:
            st.divider()
            anchor("resultado")
            st.subheader(f"Ejemplo {sel + 1}")
            item = gallery[sel]
            analyze(load_square_image(config.GALLERY_DIR / item["archivo"],
                                      config.PREPROCESS_SIZE), real_cls=item["clase_real"])
            st.button("↑ Elegir otro ejemplo", key="volver", on_click=back_to_gallery)
        target = st.session_state.pop("scroll", None)
        if target:
            scroll_to(target)
        return

    if source == "Subir una foto":
        file = st.file_uploader("Imagen JPG o PNG", type=["jpg", "jpeg", "png"])
        st.caption("En el celular, este botón también permite tomar la foto con la cámara "
                   "trasera, que suele dar mejor resultado que la cámara frontal.")
    else:
        file = st.camera_input("Cámara")
    if file is None:
        return
    try:
        img = load_square_image(file, config.PREPROCESS_SIZE)
    except Exception:  # archivo dañado o formato no soportado
        st.error("No se pudo leer la imagen. Pruebe con otra foto en formato JPG o PNG.")
        return
    st.divider()
    analyze(img)


# ------------------------------------------------------------------ cómo funciona
def tab_como_funciona() -> None:
    st.subheader("Flujo de la solución")
    st.graphviz_chart("""
    digraph {
      rankdir=TB; bgcolor="transparent"; nodesep=0.4;
      node [shape=box, style="rounded,filled", fillcolor="#e8d9c5", fontcolor="#2b1d14", fontsize=11];
      edge [color="#8a8a8a"];
      a [label="Foto de la\\nmazorca"]; b [label="Corrección de\\norientación (EXIF)"];
      c [label="Recorte cuadrado\\ny 512 px"]; d [label="Redimensionado\\na 224 px"];
      e [label="MobileNetV2\\n(ImageNet + fine-tuning)"]; f [label="Probabilidad de\\n4 estadios"];
      {rank=same; a; b; c;} {rank=same; d; e; f;}   // dos filas: legible también en celular
      a -> b -> c -> d -> e -> f;
    }""", width="stretch")
    st.markdown(
        "**Tipo de problema:** aprendizaje supervisado, clasificación de imágenes en 4 clases: "
        "sana (h0) y tres estadios de moniliasis (m1, m2, m3).\n\n"
        "**Técnica:** transfer learning sobre MobileNetV2 en dos fases (clasificador con el "
        "backbone congelado y luego fine-tuning de las últimas capas), con aumento de datos y "
        "pesos por clase.")

    eda, splits = read_json("eda_summary.json"), read_json("splits_summary.json")
    if eda and splits:
        st.subheader("Datos y limpieza")
        c1, c2, c3 = st.columns(3)
        c1.metric("Imágenes en el dataset", eda["total_imagenes"])
        c2.metric("Imágenes con etiqueta consistente", splits["imagenes_usadas"])
        c3.metric("Fotos con rotación EXIF", f"{eda['orientacion_exif'].get('6', 0) + eda['orientacion_exif'].get('3', 0) + eda['orientacion_exif'].get('8', 0)}")
        exc = splits["imagenes_excluidas"]
        ver = splits["verificacion"]
        st.markdown(
            f"- Se excluyeron **{exc.get('misma_foto_en_varias_carpetas', 0)}** imágenes copiadas "
            f"en dos carpetas de clase y **{exc.get('categoria_coco_distinta_a_carpeta', 0)}** "
            "cuya anotación no coincidía con su carpeta.\n"
            f"- Fotos casi idénticas se agruparon (pHash) para que no quedaran repartidas entre "
            f"entrenamiento y prueba: **{splits['imagenes_en_grupos_2+']}** imágenes en "
            f"**{splits['grupos_con_2+_imagenes']}** grupos.\n"
            f"- Las particiones oficiales tenían **{ver['pares_casi_duplicados_entre_particiones_oficiales']}** "
            "pares de casi-duplicados entre particiones; las propias, "
            f"**{ver['pares_casi_duplicados_entre_particiones_propias']}**.")
        counts = pd.DataFrame(splits["conteo_por_particion_y_clase"]).T
        st.dataframe(counts, width="content")
    figure("eda_samples.png", "Ejemplos por clase")


# ---------------------------------------------------------------------- desempeño
def tab_desempeno() -> None:
    test = read_json(f"eval_{config.APP_MODEL}_test.json")
    if not test:
        st.info("Métricas de test no disponibles.")
        return
    st.subheader(f"Resultado en el conjunto de test (n = {test['n']}, evaluado una sola vez)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Exactitud", pct(test["accuracy"]),
              help=f"IC 95 % bootstrap: {pct(test['accuracy_ic95'][0])}–{pct(test['accuracy_ic95'][1])}")
    c2.metric("F1-macro", f"{test['f1_macro']:.3f}",
              help=f"IC 95 %: {test['f1_macro_ic95'][0]:.3f}–{test['f1_macro_ic95'][1]:.3f}")
    c3.metric("Exactitud balanceada", f"{test['balanced_accuracy']:.3f}")

    col_a, col_b = st.columns(2)
    with col_a:
        figure(f"eval_{config.APP_MODEL}_test_confusion.png", "Matriz de confusión (test)")
    with col_b:
        per = pd.DataFrame(test["por_clase"]).T[["precision", "recall", "f1-score", "support"]]
        per.index = [f"{c} · {config.CLASS_NAMES[c]}" for c in per.index]
        st.dataframe(per.round(3), width="stretch")
        st.markdown("Los errores se concentran entre **estadios contiguos** (sana ↔ m1, "
                    "m2 ↔ m3). La debilidad principal es la detección de infecciones tempranas (m1).")

    st.subheader("Comparación de modelos (validación)")
    rows = []
    for name in ["mobilenetv2", "vgg16", "mobilenetv2_384"]:
        ev, tr = read_json(f"eval_{name}_val.json"), read_json(f"train_{name}.json")
        if ev and tr:
            rows.append({"modelo": name, "resolución": tr["img_size"],
                         "F1-macro": ev["f1_macro"], "exactitud": ev["accuracy"],
                         "recall m1": ev["por_clase"]["m1"]["recall"],
                         "ms/imagen (CPU)": ev["ms_por_imagen_cpu"],
                         "tamaño (MB)": ev["tamano_modelo_mb"], "parámetros": ev["parametros"],
                         "entrenamiento (s)": round(tr["segundos_fase1"] + tr["segundos_fase2"])})
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("modelo"), width="stretch")
    for f, label in [("compare_mobilenetv2_vs_vgg16_val.json", "MobileNetV2 vs VGG16"),
                     ("compare_mobilenetv2_vs_mobilenetv2_384_val.json", "224 px vs 384 px")]:
        comp = read_json(f)
        if comp:
            lo, hi = comp["ic95_bootstrap_pareado"]
            st.markdown(f"- **{label}:** diferencia de F1-macro = {comp['valor']:+.3f} "
                        f"(IC 95 % pareado: {lo:+.3f} a {hi:+.3f}): {comp['interpretacion']}.")
    st.markdown("Con desempeño estadísticamente equivalente se eligió **MobileNetV2 a 224 px**: "
                "6 veces más liviano y unas 3,6 veces más rápido que VGG16.")

    val = read_json(f"eval_{config.APP_MODEL}_val.json")
    if val:
        st.subheader("Confianza: cobertura frente a exactitud (validación)")
        conf = pd.DataFrame(val["analisis_confianza"]).T
        conf.index.name = "umbral"
        st.dataframe(conf, width="content")
        st.caption(f"La app usa el umbral {TH}: por debajo, muestra 'resultado incierto'.")
    figure(f"train_{config.APP_MODEL}_curves.png", "Curvas de aprendizaje")


# ---------------------------------------------------------------------- acerca de
def tab_acerca() -> None:
    st.subheader("Proyecto")
    st.markdown(f"Proyecto final del módulo *Fundamentos de Inteligencia Artificial*, "
                f"Universidad Yachay Tech.\n\n**Autores:** {AUTORES}\n\n"
                f"**Código:** [{REPO_URL}]({REPO_URL})")
    st.subheader("Datos")
    st.markdown("Imágenes de **CocoaMoniliaDataSet** (Zenodo, DOI "
                "[10.5281/zenodo.17716661](https://doi.org/10.5281/zenodo.17716661)), "
                "publicado bajo licencia **CC BY 4.0**. Las imágenes de la galería son "
                "copias redimensionadas del conjunto original.")
    st.subheader("Limitaciones")
    st.markdown(
        "- Las fotos de entrenamiento provienen de Colombia; el desempeño con cacao "
        "ecuatoriano (Nacional, CCN-51) no está validado.\n"
        "- Detecta solo una parte de las infecciones tempranas (m1): es el error más frecuente.\n"
        "- Clasifica la imagen completa: con varias mazorcas en distintos estadios, "
        "la respuesta es ambigua.\n"
        "- No reconoce imágenes fuera de su dominio (por ejemplo, otros cultivos).")


# ----------------------------------------------------------------------- principal
col_title, col_theme = st.columns([5, 1], vertical_alignment="center")
with col_title:
    st.title("🍫 Detección de moniliasis en mazorcas de cacao")
    st.caption("Clasificación del estadio de *Moniliophthora roreri* a partir de una foto")
with col_theme:
    theme_button()
t1, t2, t3, t4 = st.tabs(["Diagnóstico", "Cómo funciona", "Desempeño", "Acerca de"])
with t1:
    tab_diagnostico()
with t2:
    tab_como_funciona()
with t3:
    tab_desempeno()
with t4:
    tab_acerca()
