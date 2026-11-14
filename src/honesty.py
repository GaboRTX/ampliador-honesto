"""Capa de honestidad: qué se conservó y qué inventó el modelo.

Tres herramientas, todas con procesamiento digital clásico (sin IA):

1. Línea base bicúbica: lo que se obtiene ampliando SIN inventar nada
   (solo interpolación entre píxeles vecinos).
2. Mapa de "detalle agregado": diferencia entre el resultado de la IA y la
   bicúbica. Donde difieren, el modelo puso algo que la interpolación no
   puede deducir: es detalle INVENTADO (puede ser plausible o no).
3. Prueba de consistencia: achicamos el resultado de la IA al tamaño
   original y lo comparamos con tu imagen. Si el modelo respetó tu dato, al
   achicarlo debería volver a parecerse a tu original. Pasar esta prueba NO
   significa que el detalle sea verdadero: muchas imágenes distintas, al
   achicarse, dan la misma imagen chica.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
from skimage.filters import gaussian

FACTOR = 4
TEXTO_AVISO = "Ampliada con IA - los detalles finos fueron generados por un modelo"


# ---------------------------------------------------------------- utilidades
def ampliar_bicubica(imagen: Image.Image, factor: int = FACTOR) -> Image.Image:
    """Interpolación bicúbica: cada píxel nuevo es un promedio ponderado de 4×4 vecinos."""
    return imagen.resize((imagen.width * factor, imagen.height * factor), Image.Resampling.BICUBIC)


def luminancia(imagen: Image.Image) -> np.ndarray:
    """Pasa a luminancia (Y del espacio YCbCr, recomendación ITU-R 601) en rango 0-1.

    Usamos solo la luminancia porque el ojo es mucho más sensible al brillo
    que al color, y ahí se ve el "detalle" (bordes, letras).
    """
    rgb = np.asarray(imagen.convert("RGB"), dtype=np.float32) / 255.0
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def normalizar(mapa: np.ndarray, percentil: float = 99.0) -> np.ndarray:
    """Lleva el mapa a 0-1 usando el percentil 99 como máximo (ignora picos aislados)."""
    tope = float(np.percentile(mapa, percentil))
    if tope <= 1e-6:
        return np.zeros_like(mapa)
    return np.clip(mapa / tope, 0.0, 1.0)


def colorear_mapa(mapa: np.ndarray, base: Image.Image) -> Image.Image:
    """Superpone el mapa (0-1) como 'calor' sobre la imagen en gris.

    Negro/gris = no cambió · rojo = cambió algo · amarillo/blanco = cambió mucho.
    """
    m = np.clip(mapa, 0.0, 1.0)
    calor = np.stack(
        [np.clip(3 * m, 0, 1), np.clip(3 * m - 1, 0, 1), np.clip(3 * m - 2, 0, 1)], axis=-1
    )
    gris = luminancia(base.resize((m.shape[1], m.shape[0])))[..., None] * 0.45
    alfa = np.clip(m * 1.5, 0, 1)[..., None]
    mezcla = gris * (1 - alfa) + calor * alfa
    return Image.fromarray((mezcla * 255).astype(np.uint8))


# ------------------------------------------------------- mapa de lo inventado
@dataclass
class MapaInventado:
    imagen: Image.Image       # mapa coloreado para mostrar
    porcentaje: float         # % de la imagen con detalle agregado notable
    mapa: np.ndarray          # mapa normalizado 0-1 (para tests / análisis)


def mapa_detalle_agregado(resultado: Image.Image, bicubica: Image.Image, umbral: float = 0.25) -> MapaInventado:
    """Dónde la IA se aparta de la interpolación clásica = detalle que inventó."""
    if resultado.size != bicubica.size:
        bicubica = bicubica.resize(resultado.size, Image.Resampling.BICUBIC)
    diferencia = np.abs(luminancia(resultado) - luminancia(bicubica))
    # Filtro gaussiano (pasa-bajos): agrupa diferencias de píxeles sueltos en
    # "zonas", que son más fáciles de leer para una persona.
    suavizada = gaussian(diferencia, sigma=2)
    # Escala absoluta: 0.08 de luminancia (~20 niveles de gris) ya es un cambio visible.
    mapa = np.clip(suavizada / 0.08, 0.0, 1.0)
    porcentaje = float((mapa > umbral).mean() * 100)
    return MapaInventado(imagen=colorear_mapa(mapa, resultado), porcentaje=porcentaje, mapa=mapa)


# ------------------------------------------------------ prueba de consistencia
@dataclass
class Consistencia:
    error_medio: float        # diferencia promedio en niveles de gris (0-255)
    porcentaje_distinto: float  # % de píxeles que difieren más de 10 niveles
    imagen: Image.Image       # mapa coloreado, al tamaño del resultado
    veredicto: str


def prueba_consistencia(entrada: Image.Image, resultado: Image.Image) -> Consistencia:
    """Achica el resultado al tamaño original y lo compara con tu imagen."""
    # BOX = promedio de cada bloque de 4×4 píxeles: simula "bajar la resolución".
    reducida = resultado.resize(entrada.size, Image.Resampling.BOX)
    diferencia = np.abs(luminancia(reducida) - luminancia(entrada)) * 255
    error_medio = float(diferencia.mean())
    porcentaje = float((diferencia > 10).mean() * 100)

    if error_medio < 4:
        veredicto = "Muy consistente: al achicar el resultado, vuelve a parecerse a tu original."
    elif error_medio < 10:
        veredicto = "Bastante consistente, con algunas zonas donde el modelo se apartó de tu imagen."
    else:
        veredicto = ("Poco consistente: el modelo cambió cosas que SÍ estaban en tu imagen. "
                     "Bajá el control de 'cuánto puede inventar'.")

    # Escala absoluta: 30 niveles de gris de diferencia ya se pinta "a fondo".
    mapa = np.clip(diferencia / 30, 0.0, 1.0)
    mapa_img = colorear_mapa(mapa, reducida).resize(resultado.size, Image.Resampling.NEAREST)
    return Consistencia(error_medio, porcentaje, mapa_img, veredicto)


# ----------------------------------------------------------- aviso y exportación
def _fuente(tamano: int) -> ImageFont.ImageFont:
    for nombre in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(nombre, tamano)
        except OSError:
            continue
    return ImageFont.load_default(size=tamano)


def marcar_como_ia(imagen: Image.Image, texto: str = "Ampliada con IA") -> Image.Image:
    """Agrega una etiqueta visible en la esquina inferior izquierda."""
    marcada = imagen.convert("RGB").copy()
    dibujo = ImageDraw.Draw(marcada)
    tamano = max(12, marcada.width // 40)
    fuente = _fuente(tamano)
    margen = tamano // 2
    caja = dibujo.textbbox((0, 0), texto, font=fuente)
    ancho, alto = caja[2] - caja[0], caja[3] - caja[1]
    y = marcada.height - alto - 3 * margen
    dibujo.rectangle([margen, y, margen + ancho + 2 * margen, y + alto + 2 * margen], fill=(200, 30, 30))
    dibujo.text((2 * margen, y + margen - caja[1]), texto, font=fuente, fill=(255, 255, 255))
    return marcada


def guardar_con_aviso(imagen: Image.Image, ruta: str, detalles: str = "") -> str:
    """Guarda el PNG con etiqueta visible + metadato invisible que lo declara generado con IA."""
    metadatos = PngInfo()
    metadatos.add_text("Description", TEXTO_AVISO)
    metadatos.add_text("Software", "Ampliador Honesto (stabilityai/stable-diffusion-x4-upscaler)")
    if detalles:
        metadatos.add_text("Comment", detalles)
    marcar_como_ia(imagen).save(ruta, pnginfo=metadatos)
    return ruta
