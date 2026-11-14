"""Laboratorio de verificación: medir cuánto se acerca la IA a la verdad.

Con una captura real chica NO hay referencia: no sabemos cómo era "de verdad".
Truco honesto: partimos de una imagen nítida (la verdad), la degradamos
nosotros (la achicamos ×4), la ampliamos con bicúbica y con IA, y comparamos
las dos contra la verdad.

- PSNR (dB): parecido píxel a píxel. Más alto = más parecido.
- SSIM (0 a 1): parecido de estructura (bordes, contraste). Más alto = más parecido.

Ninguna métrica dice si algo es auténtico: miden parecido, no verdad.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

FACTOR = 4


def degradar(imagen: Image.Image, factor: int = FACTOR) -> tuple[Image.Image, Image.Image]:
    """Devuelve (verdad recortada a múltiplo de 8·factor, versión chica degradada)."""
    imagen = imagen.convert("RGB")
    paso = 8 * factor
    ancho, alto = (imagen.width // paso) * paso, (imagen.height // paso) * paso
    if ancho == 0 or alto == 0:
        raise ValueError(f"La imagen tiene que medir al menos {paso}×{paso} px.")
    verdad = imagen.crop((0, 0, ancho, alto))
    # BOX promedia bloques: simula una cámara/captura de menor resolución.
    chica = verdad.resize((ancho // factor, alto // factor), Image.Resampling.BOX)
    return verdad, chica


@dataclass
class Comparacion:
    psnr: float
    ssim: float


def comparar(verdad: Image.Image, candidata: Image.Image) -> Comparacion:
    if candidata.size != verdad.size:
        candidata = candidata.resize(verdad.size, Image.Resampling.BICUBIC)
    a = np.asarray(verdad.convert("RGB"))
    b = np.asarray(candidata.convert("RGB"))
    psnr = float(peak_signal_noise_ratio(a, b, data_range=255))
    ssim = float(structural_similarity(a, b, channel_axis=-1, data_range=255))
    return Comparacion(psnr=psnr, ssim=ssim)
