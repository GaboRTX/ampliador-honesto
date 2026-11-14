"""Pruebas de la preparación de imágenes antes del modelo (no descarga el modelo)."""

from PIL import Image

from src.diffusion import intensidad_a_noise_level, preparar_imagen


def test_png_con_transparencia_queda_rgb_con_fondo_blanco():
    transparente = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    lista, _ = preparar_imagen(transparente, lado_max=128)
    assert lista.mode == "RGB"
    assert lista.getpixel((10, 10)) == (255, 255, 255)


def test_imagen_grande_se_achica_y_avisa():
    grande = Image.new("RGB", (1000, 500), "gray")
    lista, avisos = preparar_imagen(grande, lado_max=128)
    assert max(lista.size) <= 128
    assert avisos


def test_lados_multiplos_de_ocho():
    rara = Image.new("RGB", (101, 77), "white")
    lista, _ = preparar_imagen(rara, lado_max=128)
    assert lista.width % 8 == 0 and lista.height % 8 == 0


def test_intensidad_se_traduce_y_se_limita():
    assert intensidad_a_noise_level(0) == 0
    assert intensidad_a_noise_level(1) == 200
    assert intensidad_a_noise_level(5) == 200
