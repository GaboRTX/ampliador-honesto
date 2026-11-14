"""Arma el informe en lenguaje simple que ve la persona usuaria."""

from __future__ import annotations

from src.analysis import Analisis
from src.honesty import Consistencia


def _porcentaje(valor: float) -> str:
    return f"{valor * 100:.0f}%"


def armar_informe(
    ancho_original: int,
    alto_original: int,
    ancho_final: int,
    alto_final: int,
    segundos: float,
    porcentaje_inventado: float,
    consistencia: Consistencia,
    analisis: Analisis | None,
    avisos: list[str],
) -> str:
    lineas: list[str] = []

    lineas.append("### Qué hizo el sistema")
    lineas.append(
        f"Tu imagen pasó de **{ancho_original}×{alto_original}** a **{ancho_final}×{alto_final} px** "
        f"(16 veces más píxeles) en {segundos:.1f} s. "
        "**15 de cada 16 píxeles nuevos no estaban en tu imagen: los generó el modelo.**"
    )
    for aviso in avisos:
        lineas.append(f"- ℹ️ {aviso}")

    lineas.append("\n### Qué inventó el modelo")
    lineas.append(
        f"En el **{porcentaje_inventado:.0f}%** de la imagen el modelo agregó detalle que una "
        "ampliación clásica (bicúbica) no tiene. Lo ves en rojo/amarillo en el mapa "
        "**“Dónde inventó detalle”**. Ese detalle puede ser parecido a la realidad o no: "
        "no hay forma de saberlo solo con esta imagen."
    )

    lineas.append("\n### ¿Respetó tu imagen?")
    lineas.append(
        f"{consistencia.veredicto} (diferencia promedio: {consistencia.error_medio:.1f} de 255 niveles; "
        f"{consistencia.porcentaje_distinto:.0f}% de los píxeles difiere notablemente). "
        "Ojo: que sea consistente **no prueba** que los detalles sean reales."
    )

    if analisis is not None:
        lineas.append("\n### Lo que ve un segundo modelo (SigLIP 2)")
        antes, despues = analisis.calidad_antes["nítida"], analisis.calidad_despues["nítida"]
        if despues > antes + 0.05:
            comparacion = "la versión con IA **parece más nítida** que la ampliación clásica"
        elif despues < antes - 0.05:
            comparacion = "la versión con IA **no parece más nítida** que la clásica (¿vale la pena?)"
        else:
            comparacion = "las dos versiones se ven **parecidas en nitidez**"
        lineas.append(
            f"Según SigLIP, {comparacion} (“nítida”: {_porcentaje(antes)} → {_porcentaje(despues)})."
        )
        principal_antes = analisis.contenido_principal_antes
        principal_despues = analisis.contenido_principal_despues
        lineas.append(f"Contenido detectado: **{principal_despues}**.")
        if principal_antes != principal_despues:
            lineas.append(
                f"⚠️ **Antes parecía “{principal_antes}” y ahora parece “{principal_despues}”**: "
                "el modelo cambió lo que se ve. Revisalo con cuidado."
            )
        lineas.append("_SigLIP también puede equivocarse: es una ayuda, no una prueba._")

        if analisis.riesgos:
            lineas.append("\n### ⚠️ Atención antes de publicar")
            if "dashboard con números" in analisis.riesgos or "texto / documento" in analisis.riesgos:
                lineas.append(
                    "- Tu imagen tiene **texto o números**. El modelo puede dibujar letras y cifras "
                    "que **parecen reales pero son inventadas** (un 8 que era un 3, un $ que no existe). "
                    "**Compará cada número con el dashboard original antes de publicarlo.**"
                )
            if "rostro de una persona" in analisis.riesgos:
                lineas.append(
                    "- Hay un **rostro**. El modelo puede cambiar rasgos y hacer que parezca otra persona."
                )

    lineas.append("\n### Cómo verificarlo")
    lineas.append(
        "La única forma de confirmar un detalle es **con la fuente original**: volver a sacar la "
        "captura en alta resolución o mirar el dashboard. Si no podés verificarlo, publicá la imagen "
        "aclarando que fue **ampliada con IA** (la descarga ya lleva esa etiqueta)."
    )
    return "\n".join(lineas)
