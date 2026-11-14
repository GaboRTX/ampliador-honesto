# Instrucciones para agentes de IA

Este archivo le da contexto a cualquier agente de código (Claude, Codex, Copilot, etc.) que trabaje en este repositorio.

## Proyecto
**Ampliador Honesto**: app local que amplía ×4 capturas de pantalla chicas (dashboards, gráficos) con un modelo de difusión y le muestra a la persona usuaria **qué parte del resultado inventó el modelo**.

## Stack
- Python 3.12 + `uv`
- Interfaz: Gradio
- Difusión: `stabilityai/stable-diffusion-x4-upscaler` (diffusers)
- Análisis visual: `google/siglip2-base-patch16-224` (zero-shot, transformers)
- Procesamiento: Pillow, NumPy, scikit-image

## Comandos
```bash
uv sync                 # instalar dependencias
uv run app.py           # levantar la app en http://127.0.0.1:7860
uv run app.py --share   # link público (para Colab)
uv run pytest           # pruebas (no descargan modelos)
```

## Estructura
- `app.py`: solo interfaz. No poner lógica de procesamiento acá.
- `src/diffusion.py`: carga del modelo y ampliación.
- `src/honesty.py`: línea base bicúbica, mapas de diferencias, avisos, marca de agua.
- `src/analysis.py`: análisis con SigLIP 2.
- `src/metrics.py`: PSNR / SSIM para el laboratorio de verificación.

## Reglas
1. Los modelos se cargan **una sola vez** (función `cargar_modelo()` con caché).
2. Todo texto que ve la persona usuaria va **en español simple, sin jerga**.
3. Los errores se muestran con `gr.Error` y una sugerencia de qué hacer.
4. Nunca decir que la IA "recupera" detalle: decir que lo **reconstruye** o lo **inventa**.
5. Toda imagen exportada lleva marca visible y metadato "Ampliada con IA".
6. Las funciones de `honesty.py` y `metrics.py` no dependen de modelos y tienen tests en `tests/`.
7. Mostrar el plan antes de escribir código y hacer cambios chicos (un cambio = un commit).
8. No subir tokens, `.env` ni cachés de modelos.
