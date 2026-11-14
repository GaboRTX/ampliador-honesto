"""Pruebas de la capa de honestidad y las métricas (no descargan modelos)."""

from pathlib import Path

import numpy as np
from PIL import Image

from src import honesty, metrics

EJEMPLOS = Path(__file__).resolve().parent.parent / "assets" / "ejemplos"


def imagen_ejemplo(nombre: str) -> Image.Image:
    return Image.open(EJEMPLOS / nombre).convert("RGB")


def test_bicubica_amplia_cuatro_veces():
    chica = imagen_ejemplo("dashboard_chico.png")
    grande = honesty.ampliar_bicubica(chica)
    assert grande.size == (chica.width * 4, chica.height * 4)


def test_mapa_vacio_si_no_hay_invento():
    """Si la 'IA' devolviera exactamente la bicúbica, no inventó nada."""
    bicubica = honesty.ampliar_bicubica(imagen_ejemplo("dashboard_chico.png"))
    mapa = honesty.mapa_detalle_agregado(bicubica, bicubica)
    assert mapa.porcentaje == 0
    assert mapa.imagen.size == bicubica.size


def test_mapa_detecta_detalle_agregado():
    """La imagen nítida tiene detalle que la bicúbica no tiene: el mapa lo marca."""
    nitida = imagen_ejemplo("dashboard_nitido.png")
    bicubica = honesty.ampliar_bicubica(imagen_ejemplo("dashboard_chico.png"))
    mapa = honesty.mapa_detalle_agregado(nitida, bicubica)
    assert mapa.porcentaje > 1


def test_consistencia_alta_para_la_verdad():
    """La imagen nítida real, achicada, vuelve a ser exactamente la chica."""
    chica = imagen_ejemplo("dashboard_chico.png")
    nitida = imagen_ejemplo("dashboard_nitido.png")
    resultado = honesty.prueba_consistencia(chica, nitida)
    assert resultado.error_medio < 1
    assert resultado.veredicto.startswith("Muy consistente")


def test_consistencia_baja_si_cambia_el_contenido():
    chica = imagen_ejemplo("dashboard_chico.png")
    otra = imagen_ejemplo("texto_nitido.png")
    resultado = honesty.prueba_consistencia(chica, otra)
    assert resultado.error_medio > 10


def test_marca_y_metadato(tmp_path):
    ruta = tmp_path / "salida.png"
    honesty.guardar_con_aviso(imagen_ejemplo("grafico_nitido.png"), str(ruta), "prueba")
    guardada = Image.open(ruta)
    assert "IA" in guardada.text["Description"]
    # La esquina inferior izquierda tiene la etiqueta roja.
    pixel = np.asarray(guardada.convert("RGB"))[guardada.height - 12, 12]
    assert pixel[0] > 150 and pixel[1] < 80


def test_metricas_verdad_contra_si_misma():
    verdad = imagen_ejemplo("grafico_nitido.png")
    comparacion = metrics.comparar(verdad, verdad)
    assert comparacion.ssim > 0.999


def test_degradar_achica_cuatro_veces():
    verdad, chica = metrics.degradar(imagen_ejemplo("texto_nitido.png"))
    assert chica.size == (verdad.width // 4, verdad.height // 4)
