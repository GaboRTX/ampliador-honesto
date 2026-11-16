"""Ampliación ×4 con el modelo de difusión Stable Diffusion x4 Upscaler.

Idea clave para la defensa:
- El modelo NO recupera píxeles perdidos. Parte de ruido y lo va "limpiando"
  (difusión inversa) guiado por dos cosas: la imagen chica original y un texto.
- `noise_level` controla cuánto ruido se le agrega a la imagen chica antes de
  usarla como guía. Más ruido = el modelo confía menos en tu imagen y
  INVENTA más. Por eso lo exponemos como "cuánto puede inventar el modelo".
"""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import torch
from PIL import Image

registro = logging.getLogger(__name__)

MODELO_ID = "stabilityai/stable-diffusion-x4-upscaler"
FACTOR = 4

HAY_GPU = torch.cuda.is_available()
DISPOSITIVO = "cuda" if HAY_GPU else "cpu"

# Lado máximo de la imagen de ENTRADA. La salida es 4 veces más grande.
# En CPU usamos imágenes chicas para que no tarde una eternidad.
LADO_MAX_ENTRADA = 512 if HAY_GPU else 128
PASOS = 25 if HAY_GPU else 15

# Presets pensados para la persona usuaria (no tiene que escribir prompts).
# Los prompts van en inglés porque el modelo se entrenó con textos en inglés.
PRESETS: dict[str, str] = {
    "Captura de dashboard": "a sharp, clean screenshot of a business intelligence dashboard, charts, crisp text",
    "Gráfico o diagrama": "a sharp, clean chart with crisp lines and labels, flat design",
    "Foto": "a sharp, detailed, high quality photograph",
}
PROMPT_NEGATIVO = "blurry, noisy, distorted, jpeg artifacts, deformed text"


@dataclass
class ResultadoAmpliacion:
    """Todo lo que devuelve una ampliación, para poder explicarla después."""

    entrada: Image.Image          # la imagen que realmente vio el modelo
    resultado: Image.Image        # la imagen ampliada por la IA
    segundos: float
    noise_level: int
    pasos: int
    avisos: list[str] = field(default_factory=list)


def intensidad_a_noise_level(intensidad: float) -> int:
    """Traduce el deslizador (0 a 1) al parámetro técnico noise_level (0 a 200).

    0   -> el modelo se pega lo más posible a tu imagen
    1   -> el modelo se toma mucha libertad (inventa más)
    """
    intensidad = min(max(float(intensidad), 0.0), 1.0)
    return int(round(intensidad * 200))


def preparar_imagen(imagen: Image.Image, lado_max: int = LADO_MAX_ENTRADA) -> tuple[Image.Image, list[str]]:
    """Convierte a RGB, achica si hace falta y ajusta a múltiplos de 8.

    Devuelve la imagen lista y una lista de avisos para la persona usuaria.
    """
    avisos: list[str] = []

    # PNG con transparencia: la pegamos sobre fondo blanco (si no, queda negro).
    if imagen.mode in ("RGBA", "LA", "P"):
        imagen = imagen.convert("RGBA")
        fondo = Image.new("RGB", imagen.size, (255, 255, 255))
        fondo.paste(imagen, mask=imagen.split()[-1])
        imagen = fondo
    else:
        imagen = imagen.convert("RGB")

    ancho, alto = imagen.size
    if max(ancho, alto) > lado_max:
        imagen = imagen.copy()
        # thumbnail usa un filtro anti-aliasing al achicar (evita "escalones").
        imagen.thumbnail((lado_max, lado_max), Image.Resampling.LANCZOS)
        avisos.append(
            f"Tu imagen era de {ancho}×{alto} px. La achiqué a {imagen.width}×{imagen.height} px "
            f"antes de ampliarla, porque en este equipo el límite de entrada es {lado_max} px."
        )

    # La red interna (U-Net) reduce la imagen 3 veces a la mitad: los lados
    # tienen que ser múltiplos de 8. Recortamos lo mínimo necesario.
    ancho8, alto8 = (imagen.width // 8) * 8, (imagen.height // 8) * 8
    if ancho8 < 16 or alto8 < 16:
        raise ValueError("La imagen es demasiado chica (mínimo 16×16 px).")
    if (ancho8, alto8) != imagen.size:
        imagen = imagen.crop((0, 0, ancho8, alto8))

    return imagen, avisos


@lru_cache(maxsize=1)
def cargar_modelo():
    """Carga el pipeline UNA sola vez (lru_cache guarda el resultado)."""
    from diffusers import StableDiffusionUpscalePipeline

    registro.info("Cargando %s en %s...", MODELO_ID, DISPOSITIVO)
    tipo = torch.float16 if HAY_GPU else torch.float32
    pipe = StableDiffusionUpscalePipeline.from_pretrained(MODELO_ID, torch_dtype=tipo)
    pipe = pipe.to(DISPOSITIVO)

    # Optimizaciones de memoria: calculan la atención y el decodificador
    # por partes. Un poco más lento, pero entra en GPUs chicas.
    pipe.enable_attention_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _argumentos_de_progreso(pipe, pasos: int, al_avanzar: Callable[[float], None] | None) -> dict:
    """Arma el callback de progreso según la versión de diffusers instalada."""
    if al_avanzar is None:
        return {}
    parametros = inspect.signature(pipe.__call__).parameters
    if "callback_on_step_end" in parametros:
        def cb(_pipe, paso, _t, kwargs):
            al_avanzar((paso + 1) / pasos)
            return kwargs
        return {"callback_on_step_end": cb}
    if "callback" in parametros:
        return {"callback": lambda paso, _t, _lat: al_avanzar((paso + 1) / pasos), "callback_steps": 1}
    return {}


def ampliar(
    imagen: Image.Image,
    intensidad: float = 0.1,
    preset: str = "Captura de dashboard",
    al_avanzar: Callable[[float], None] | None = None,
    semilla: int = 42,
    descripcion: str = "",
) -> ResultadoAmpliacion:
    """Amplía la imagen ×4 con difusión.

    La semilla fija hace que el resultado sea reproducible: misma imagen +
    mismos controles = mismo resultado. Si la cambiás, el modelo inventa
    OTROS detalles (¡prueba de que son inventados!).
    """
    entrada, avisos = preparar_imagen(imagen)
    noise_level = intensidad_a_noise_level(intensidad)
    prompt = PRESETS.get(preset, PRESETS["Captura de dashboard"])
    # La descripción de la persona usuaria va primero: es lo que más pesa.
    # Ojo: el texto GUÍA lo que el modelo inventa (si decís "gato", dibuja pelo de gato).
    if descripcion and descripcion.strip():
        prompt = f"{descripcion.strip()}, {prompt}"
    pipe = cargar_modelo()
    generador = torch.Generator(device=DISPOSITIVO).manual_seed(int(semilla))

    inicio = time.perf_counter()
    try:
        salida = pipe(
            prompt=prompt,
            negative_prompt=PROMPT_NEGATIVO,
            image=entrada,
            noise_level=noise_level,
            num_inference_steps=PASOS,
            guidance_scale=7.5,
            generator=generador,
            **_argumentos_de_progreso(pipe, PASOS, al_avanzar),
        )
    except torch.cuda.OutOfMemoryError:
        # Plan B automático: probamos con la mitad del tamaño.
        torch.cuda.empty_cache()
        lado = max(entrada.size) // 2
        if lado < 64:
            raise
        avisos.append("Se quedó sin memoria de video: reintenté con la imagen a la mitad de tamaño.")
        entrada, _ = preparar_imagen(entrada, lado_max=lado)
        salida = pipe(
            prompt=prompt,
            negative_prompt=PROMPT_NEGATIVO,
            image=entrada,
            noise_level=noise_level,
            num_inference_steps=PASOS,
            guidance_scale=7.5,
            generator=generador,
        )
    segundos = time.perf_counter() - inicio

    resultado = salida.images[0].convert("RGB")
    # Por seguridad, forzamos exactamente 4× (el VAE a veces redondea).
    tamano_esperado = (entrada.width * FACTOR, entrada.height * FACTOR)
    if resultado.size != tamano_esperado:
        resultado = resultado.resize(tamano_esperado, Image.Resampling.LANCZOS)

    return ResultadoAmpliacion(
        entrada=entrada,
        resultado=resultado,
        segundos=segundos,
        noise_level=noise_level,
        pasos=PASOS,
        avisos=avisos,
    )
