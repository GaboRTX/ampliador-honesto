"""Análisis visual con SigLIP 2 (clasificación zero-shot).

SigLIP compara una imagen con frases de texto y da un puntaje de qué tan
bien encaja cada frase. "Zero-shot" = no lo entrenamos para esto: le damos
las frases en el momento.

Lo usamos para dos cosas:
1. CALIDAD: ¿la versión IA "parece" más nítida que la bicúbica?
   (comparamos las dos, nunca miramos un número solo).
2. CONTENIDO / RIESGO: ¿hay texto, números o caras? Ahí inventar es peligroso
   (una cifra inventada en un dashboard es un dato falso).

Ojo: SigLIP no fue entrenado para medir calidad y se equivoca. Es una
ayuda, no una prueba.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from PIL import Image

MODELO_ANALISIS = "google/siglip2-base-patch16-224"

# frase en inglés (lo que entiende mejor el modelo) -> etiqueta en español
CALIDAD = {
    "a sharp, crisp, high resolution image": "nítida",
    "a blurry, out of focus image": "borrosa",
    "a pixelated, low resolution image": "pixelada",
}
CONTENIDO = {
    "a screenshot of a dashboard with charts and numbers": "dashboard con números",
    "a document or slide with a lot of text": "texto / documento",
    "a photo of a person's face": "rostro de una persona",
    "a photo of a landscape or objects": "foto de paisaje u objetos",
}
ETIQUETAS_DE_RIESGO = {"dashboard con números", "texto / documento", "rostro de una persona"}


@lru_cache(maxsize=1)
def cargar_clasificador():
    """Carga SigLIP 2 una sola vez."""
    from transformers import pipeline

    from src.diffusion import DISPOSITIVO

    return pipeline("zero-shot-image-classification", model=MODELO_ANALISIS, device=DISPOSITIVO)


def _puntajes(imagen: Image.Image, frases: dict[str, str]) -> dict[str, float]:
    clasificador = cargar_clasificador()
    salida = clasificador(imagen, candidate_labels=list(frases), hypothesis_template="{}")
    crudos = {frases[s["label"]]: float(s["score"]) for s in salida}
    # SigLIP da un puntaje independiente por frase (sigmoide), que suele ser
    # muy chico. Lo pasamos a proporción relativa (suman 1) para comparar.
    total = sum(crudos.values()) or 1.0
    return {etiqueta: valor / total for etiqueta, valor in crudos.items()}


def recorte_central(imagen: Image.Image, lado: int = 224) -> Image.Image:
    """Recorte del centro a resolución nativa. Si achicáramos la imagen entera
    a 224 px (lo que hace SigLIP), la diferencia de nitidez desaparecería."""
    lado = min(lado, imagen.width, imagen.height)
    x = (imagen.width - lado) // 2
    y = (imagen.height - lado) // 2
    return imagen.crop((x, y, x + lado, y + lado))


@dataclass
class Analisis:
    calidad_antes: dict[str, float]
    calidad_despues: dict[str, float]
    contenido_antes: dict[str, float]
    contenido_despues: dict[str, float]
    riesgos: list[str]

    @property
    def contenido_principal_antes(self) -> str:
        return max(self.contenido_antes, key=self.contenido_antes.get)

    @property
    def contenido_principal_despues(self) -> str:
        return max(self.contenido_despues, key=self.contenido_despues.get)


def analizar(bicubica: Image.Image, resultado: Image.Image) -> Analisis:
    """Compara la ampliación clásica (antes) con la de la IA (después)."""
    calidad_antes = _puntajes(recorte_central(bicubica), CALIDAD)
    calidad_despues = _puntajes(recorte_central(resultado), CALIDAD)
    contenido_antes = _puntajes(bicubica, CONTENIDO)
    contenido_despues = _puntajes(resultado, CONTENIDO)

    riesgos = [
        etiqueta for etiqueta in ETIQUETAS_DE_RIESGO
        if contenido_antes.get(etiqueta, 0) >= 0.30 or contenido_despues.get(etiqueta, 0) >= 0.30
    ]
    return Analisis(calidad_antes, calidad_despues, contenido_antes, contenido_despues, sorted(riesgos))
