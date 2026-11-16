"""Ampliador Honesto: interfaz web local (Gradio).

Ejecutar:  uv run app.py            (local)
           uv run app.py --share    (link público, para Colab)

Este archivo solo arma la interfaz. La lógica está en src/.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile

import gradio as gr
import numpy as np
from PIL import Image, UnidentifiedImageError

from src import diffusion, honesty, metrics
from src.analysis import analizar, cargar_clasificador
from src.informe import armar_informe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
registro = logging.getLogger("ampliador")

MAX_MB = 10
CARPETA_SALIDAS = os.path.join(tempfile.gettempdir(), "ampliador_honesto")
os.makedirs(CARPETA_SALIDAS, exist_ok=True)
CARPETA_EJEMPLOS = os.path.join(os.path.dirname(__file__), "assets", "ejemplos")

AVISO_PRINCIPAL = (
    "⚠️ **Esta herramienta no “recupera” tu imagen: la reconstruye.** "
    "Un modelo de IA dibuja los detalles que faltan para que se vea nítida. "
    "Lo que ves puede ser **creíble pero inventado**, sobre todo en **números, letras y caras**. "
    "Revisá siempre contra el original antes de publicar."
)

AVISO_EQUIPO = (
    f"🖥️ Corriendo en **{'GPU' if diffusion.HAY_GPU else 'CPU (sin placa de video)'}**: "
    f"entrada máxima {diffusion.LADO_MAX_ENTRADA} px, {diffusion.PASOS} pasos. "
    + ("Cada imagen tarda unos segundos." if diffusion.HAY_GPU
       else "Cada imagen puede tardar **varios minutos**. Para ir más rápido, usá la versión de Colab.")
)


# ------------------------------------------------------------------ helpers
def abrir_imagen(ruta: str | None) -> Image.Image:
    """Valida y abre la imagen, con errores en lenguaje humano."""
    if not ruta:
        raise gr.Error("Primero subí una imagen o elegí un ejemplo de abajo.")
    megas = os.path.getsize(ruta) / (1024 * 1024)
    if megas > MAX_MB:
        raise gr.Error(f"La imagen pesa {megas:.1f} MB. Probá con una de menos de {MAX_MB} MB.")
    try:
        imagen = Image.open(ruta)
        imagen.load()
    except (UnidentifiedImageError, OSError):
        raise gr.Error("No pude abrir ese archivo como imagen. Probá con un PNG o JPG.") from None
    return imagen


def ruta_salida(nombre: str) -> str:
    return os.path.join(CARPETA_SALIDAS, nombre)


def avance(progreso: gr.Progress, desde: float, hasta: float, texto: str):
    """Convierte el avance del modelo (0-1) en la barra de progreso de Gradio."""
    return lambda fraccion: progreso(desde + (hasta - desde) * fraccion, desc=texto)


# --------------------------------------------------------- pestaña principal
def procesar(ruta, intensidad, preset, usar_analisis, descripcion="", semilla=42, progreso=gr.Progress()):
    imagen = abrir_imagen(ruta)
    ancho_original, alto_original = imagen.size

    progreso(0.02, desc="Preparando la imagen...")
    try:
        ampliada = diffusion.ampliar(
            imagen, intensidad, preset,
            al_avanzar=avance(progreso, 0.05, 0.80,
                              "Reconstruyendo detalles con el modelo de difusión (dibujando lo que falta)..."),
            semilla=int(semilla or 0),
            descripcion=descripcion or "",
        )
    except ValueError as error:
        raise gr.Error(str(error)) from error
    except Exception as error:  # noqa: BLE001 - mostramos un mensaje humano
        registro.exception("Falló la ampliación")
        raise gr.Error("No pude ampliar la imagen. Probá con una más chica o en formato PNG/JPG.") from error

    progreso(0.82, desc="Comparando con una ampliación clásica (bicúbica)...")
    bicubica = honesty.ampliar_bicubica(ampliada.entrada)
    inventado = honesty.mapa_detalle_agregado(ampliada.resultado, bicubica)
    consistencia = honesty.prueba_consistencia(ampliada.entrada, ampliada.resultado)

    analisis = None
    if usar_analisis:
        progreso(0.90, desc="Pidiéndole a un segundo modelo (SigLIP) que revise el resultado...")
        try:
            analisis = analizar(bicubica, ampliada.resultado)
        except Exception:  # noqa: BLE001 - el análisis es una ayuda, no debe romper la app
            registro.exception("Falló el análisis con SigLIP")
            ampliada.avisos.append("No pude correr el análisis con SigLIP esta vez.")

    informe = armar_informe(
        ancho_original, alto_original, *ampliada.resultado.size, ampliada.segundos,
        inventado.porcentaje, consistencia, analisis, ampliada.avisos,
    )

    detalles = (f"modelo={diffusion.MODELO_ID}; noise_level={ampliada.noise_level}; "
                f"pasos={ampliada.pasos}; preset={preset}; semilla={int(semilla or 0)}; "
                f"descripcion={descripcion or '-'}")
    archivo = honesty.guardar_con_aviso(ampliada.resultado, ruta_salida("ampliada_con_ia.png"), detalles)

    progreso(1.0, desc="Listo")
    return (bicubica, ampliada.resultado), inventado.imagen, consistencia.imagen, informe, archivo


# ---------------------------------------------- pestaña laboratorio (verificable)
def laboratorio(ruta, intensidad, preset, progreso=gr.Progress()):
    imagen = abrir_imagen(ruta).convert("RGB")
    # Limitamos la "verdad" para que la versión chica entre en el modelo sin achicarse de nuevo.
    imagen.thumbnail((diffusion.LADO_MAX_ENTRADA * 4, diffusion.LADO_MAX_ENTRADA * 4), Image.Resampling.LANCZOS)
    try:
        verdad, chica = metrics.degradar(imagen)
    except ValueError as error:
        raise gr.Error(str(error)) from error

    progreso(0.05, desc="Achiqué tu imagen ×4. Ahora la IA intenta reconstruirla...")
    ampliada = diffusion.ampliar(
        chica, intensidad, preset,
        al_avanzar=avance(progreso, 0.05, 0.9, "Reconstruyendo con el modelo de difusión..."),
    )
    bicubica = honesty.ampliar_bicubica(chica)
    m_bicubica = metrics.comparar(verdad, bicubica)
    m_ia = metrics.comparar(verdad, ampliada.resultado)

    error_ia = np.abs(honesty.luminancia(ampliada.resultado.resize(verdad.size)) - honesty.luminancia(verdad)) * 255
    mapa_error = honesty.colorear_mapa(np.clip(error_ia / 30, 0, 1), verdad)

    gana_psnr = "IA" if m_ia.psnr > m_bicubica.psnr else "bicúbica"
    tabla = (
        "| Método | PSNR (dB) ↑ | SSIM ↑ |\n|---|---|---|\n"
        f"| Bicúbica (sin inventar) | {m_bicubica.psnr:.2f} | {m_bicubica.ssim:.3f} |\n"
        f"| Difusión (IA) | {m_ia.psnr:.2f} | {m_ia.ssim:.3f} |\n\n"
        f"**Más parecida a la verdad píxel a píxel: {gana_psnr}.**\n\n"
        "Cómo leerlo: si la IA **se ve** más nítida pero tiene PSNR/SSIM **más bajos** que la bicúbica, "
        "esa nitidez es **inventada**: no coincide con la imagen real. "
        "El mapa muestra dónde la IA se alejó de la verdad (rojo/amarillo = se equivocó más). "
        "Fijate especialmente en los números y las letras.\n\n"
        "_Estas métricas miden parecido, no autenticidad._"
    )
    chica_vista = chica.resize(verdad.size, Image.Resampling.NEAREST)
    galeria = [
        (verdad, "1. Verdad (tu imagen original)"),
        (chica_vista, f"2. Lo que recibe el modelo ({chica.width}×{chica.height} px)"),
        (bicubica, "3. Ampliación clásica (bicúbica)"),
        (ampliada.resultado, "4. Ampliación con IA"),
    ]
    return galeria, mapa_error, tabla


# ---------------------------------------------------------------- interfaz
def ejemplos(*nombres: str) -> list[str]:
    rutas = [os.path.join(CARPETA_EJEMPLOS, n) for n in nombres]
    return [r for r in rutas if os.path.exists(r)]


def comparador(label: str):
    """gr.ImageSlider existe en Gradio 5; si no, mostramos una galería."""
    if hasattr(gr, "ImageSlider"):
        return gr.ImageSlider(label=label, type="pil")
    return gr.Gallery(label=label, columns=2)


EXPLICACION = """
## Qué es real y qué es inventado

Cuando ampliás una imagen ×4, cada píxel original se convierte en 16. **La información
de esos 15 píxeles nuevos no existe en tu imagen.** Hay dos formas de completarla:

| Método | Qué hace | Inventa |
|---|---|---|
| **Bicúbica** (clásica) | Promedia los píxeles vecinos | Nada: por eso se ve borrosa |
| **Difusión** (IA) | Dibuja detalle que *encaja* con lo que aprendió de millones de imágenes | Sí: bordes, letras, texturas |

**Escalera de la confianza:** ficticio → plausible → **creíble** → verificable.
Una imagen ampliada con IA casi nunca pasa de *creíble*. Para que sea *verificable*
necesitás evidencia externa: la captura original en alta resolución o el dashboard.

**Cómo leer las herramientas:**
- **Antes / Después**: bicúbica (izquierda) vs IA (derecha), mismo tamaño.
- **Dónde inventó detalle**: diferencia entre IA y bicúbica. Rojo/amarillo = el modelo agregó algo.
- **¿Respetó tu imagen?**: achicamos el resultado y lo comparamos con tu original. Rojo = el
  modelo cambió algo que sí estaba.
- **Control “cuánto puede inventar”**: más alto = el modelo confía menos en tu imagen.
- **Laboratorio**: con una imagen nítida, medimos cuánto se acerca la IA a la verdad.

**Nunca uses una imagen ampliada con IA como prueba, documento o fuente de datos.**
"""


with gr.Blocks(title="Ampliador Honesto") as demo:
    gr.Markdown("# 🔍 Ampliador Honesto\nAmpliá capturas chicas de dashboards y gráficos ×4, "
                "y mirá **qué parte del resultado inventó la IA**.")
    gr.Markdown(AVISO_PRINCIPAL)
    gr.Markdown(AVISO_EQUIPO)

    with gr.Tab("Ampliar mi captura"):
        with gr.Row():
            with gr.Column(scale=1):
                entrada = gr.Image(type="filepath", label="Tu captura (PNG o JPG, idealmente chica)")
                preset = gr.Radio(list(diffusion.PRESETS), value="Captura de dashboard", label="¿Qué tipo de imagen es?")
                intensidad = gr.Slider(
                    0.0, 1.0, value=0.1, step=0.05,
                    label="Cuánto puede inventar el modelo",
                    info="0 = se pega a tu imagen · 1 = se toma mucha libertad (más nítido, más invento)",
                )
                descripcion = gr.Textbox(
                    label="Opcional: ¿qué hay en la imagen?",
                    placeholder="En inglés funciona mejor. Ej.: a cat on a couch",
                    info="El texto guía lo que el modelo inventa: si decís 'cat', dibuja pelo de gato.",
                )
                with gr.Row():
                    semilla = gr.Number(value=42, precision=0, label="Semilla",
                                        info="Cambiala y el modelo inventa OTROS detalles")
                    otra = gr.Button("🎲 Otra variante")
                usar_analisis = gr.Checkbox(value=True, label="Revisar el resultado con un segundo modelo (SigLIP)")
                boton = gr.Button("Ampliar ×4", variant="primary")
                gr.Examples(
                    ejemplos("dashboard_chico.png", "grafico_chico.png", "texto_chico.png"),
                    inputs=entrada, label="Ejemplos",
                )
            with gr.Column(scale=2):
                slider = comparador("Antes (bicúbica, sin IA) / Después (IA) — arrastrá para comparar")
                with gr.Row():
                    mapa_inventado = gr.Image(type="pil", label="Dónde inventó detalle (vs. bicúbica)")
                    mapa_consistencia = gr.Image(type="pil", label="¿Respetó tu imagen? (rojo = cambió lo original)")
                informe = gr.Markdown()
                descarga = gr.File(label="Descargar (incluye etiqueta “Ampliada con IA”)")
        entradas = [entrada, intensidad, preset, usar_analisis, descripcion, semilla]
        salidas = [slider, mapa_inventado, mapa_consistencia, informe, descarga]
        boton.click(procesar, entradas, salidas)
        # "Otra variante": cambia la semilla al azar y vuelve a ampliar.
        otra.click(lambda: int(np.random.randint(0, 100_000)), None, semilla).then(procesar, entradas, salidas)

    with gr.Tab("Laboratorio: ¿cuánto se acerca a la verdad?"):
        gr.Markdown(
            "Subí una captura **nítida**. La achico ×4, la IA la reconstruye y comparo el resultado "
            "con tu original. Así podés ver, con evidencia, **qué inventa el modelo**."
        )
        with gr.Row():
            with gr.Column(scale=1):
                entrada_lab = gr.Image(type="filepath", label="Captura nítida (la 'verdad')")
                preset_lab = gr.Radio(list(diffusion.PRESETS), value="Captura de dashboard", label="Tipo de imagen")
                intensidad_lab = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Cuánto puede inventar el modelo")
                boton_lab = gr.Button("Degradar y reconstruir", variant="primary")
                gr.Examples(ejemplos("dashboard_nitido.png", "grafico_nitido.png", "texto_nitido.png"),
                            inputs=entrada_lab, label="Ejemplos")
            with gr.Column(scale=2):
                galeria_lab = gr.Gallery(label="Verdad → degradada → bicúbica → IA", columns=2, height="auto")
                mapa_lab = gr.Image(type="pil", label="Dónde la IA se alejó de la verdad")
                tabla_lab = gr.Markdown()
        boton_lab.click(laboratorio, [entrada_lab, intensidad_lab, preset_lab], [galeria_lab, mapa_lab, tabla_lab])

    with gr.Tab("Cómo funciona"):
        gr.Markdown(EXPLICACION)


def main():
    parser = argparse.ArgumentParser(description="Ampliador Honesto")
    parser.add_argument("--share", action="store_true", help="Crear un link público (útil en Colab)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--sin-precarga", action="store_true", help="No cargar los modelos al iniciar")
    args = parser.parse_args()

    if not args.sin_precarga:
        registro.info("Cargando modelos (la primera vez se descargan, ~2 GB; puede tardar)...")
        diffusion.cargar_modelo()
        cargar_clasificador()
        registro.info("Modelos listos.")

    demo.queue().launch(
        share=args.share,
        server_port=args.port,
        allowed_paths=[CARPETA_SALIDAS],
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
