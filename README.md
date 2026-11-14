# 🔍 Ampliador Honesto

> Amplía ×4 capturas chicas de dashboards y gráficos con un modelo de difusión, y le muestra a quien lo usa **qué parte del resultado inventó la IA**.

## Descripción

Quienes publican contenido educativo sobre datos (tips de BI, gráficos, KPIs) muchas veces solo tienen capturas **chicas o pixeladas**: una imagen que les mandaron por WhatsApp, un recorte de una presentación vieja o una captura hecha en una pantalla de baja resolución. Al ampliarlas para Instagram se ven borrosas.

**Ampliador Honesto** usa el modelo de difusión `stabilityai/stable-diffusion-x4-upscaler` para ampliarlas ×4 con aspecto nítido. Pero un modelo de difusión **no recupera** información: la **dibuja**. En un dashboard eso es grave, porque **un número inventado es un dato falso**. Por eso la app no solo amplía: compara contra una ampliación clásica (bicúbica), marca en un mapa **dónde inventó detalle**, verifica si el resultado respeta la imagen original, usa un segundo modelo (SigLIP 2) para detectar texto, números o caras, y exporta la imagen con una etiqueta **“Ampliada con IA”**.

Lo que la hace diferente de las apps de “mejorar foto”: está diseñada para mostrar la **diferencia entre creíble y verificable**, no para esconderla.

## Persona Usuaria

```
Nombre: Tomás
Edad: 30-40 años
Ocupación: Analista de BI en una empresa; los fines de semana publica tips de datos
           y dashboards en Instagram
Contexto tecnológico: Usa Power BI y Excel todos los días. No sabe nada de modelos
                      generativos ni de Python.

Problema actual:
"Tengo capturas viejas de dashboards que quiero usar en mis posteos, pero son chiquitas.
Cuando las agrando en Canva se ven borrosas y quedan poco profesionales."

Solución actual:
Agranda con Canva o con apps de "mejorar imagen" del celular. O rehace el gráfico de
cero, lo que le lleva horas.

Frustraciones:
- Las imágenes ampliadas se ven borrosas
- Las apps de IA dejan los números "lindos", pero no sabe si siguen siendo los mismos
- Si publica una cifra mal, pierde credibilidad frente a su audiencia

Objetivos:
- Publicar capturas nítidas en minutos
- Estar seguro de que los datos que muestra son los reales
- Éxito: la imagen se ve profesional y ningún seguidor le marca un número equivocado

Contexto de uso:
En su notebook, el domingo a la tarde, mientras arma los posteos de la semana.
```

**Declaración del proyecto:** Mi sistema permite que Tomás (analista de BI que publica contenido en Instagram) pueda ampliar capturas chicas de dashboards aplicando el upscaler ×4 de Stable Diffusion y verificando los resultados con SigLIP 2, una comparación con interpolación bicúbica y una prueba de consistencia. Le advierte que las letras y los números ampliados son una reconstrucción del modelo y que tiene que compararlos con la fuente antes de publicar.

## Demo

🎥 **Video (5 min):** [COMPLETAR: link a YouTube/Drive, público]

![Pantalla principal](docs/capturas/principal.png)
![Mapa de lo inventado](docs/capturas/mapa.png)
![Laboratorio de verificación](docs/capturas/laboratorio.png)

> [COMPLETAR: sacar las capturas con la app corriendo y guardarlas en `docs/capturas/` con esos nombres]

## Qué es real y qué es inventado

**Qué hace el modelo y qué no puede recuperar.** Al ampliar ×4, cada píxel se convierte en 16: **15 de cada 16 píxeles no existían**. La interpolación bicúbica los completa promediando vecinos (no inventa, por eso se ve borrosa). El modelo de difusión los completa con detalle *plausible* aprendido de millones de imágenes. En una captura de 128 px, un número de 5 px de alto **no tiene información suficiente** para saber qué dígito era: el modelo dibuja *un* dígito nítido, que puede no ser el original.

**Cómo se ve en la app.** (1) Comparador antes/después: bicúbica vs IA, al mismo tamaño. (2) Mapa **“Dónde inventó detalle”**: diferencia entre IA y bicúbica (rojo/amarillo = detalle agregado), con el porcentaje de la imagen afectado. (3) **Prueba de consistencia**: achica el resultado ×4 y lo compara con el original; si el modelo cambió algo que sí estaba, aparece en rojo. (4) **Laboratorio**: con una imagen nítida conocida, degrada, reconstruye y mide (PSNR/SSIM) cuánto se acercó la IA a la verdad.

![Qué es real y qué es inventado](docs/capturas/real_inventado.png)

**Qué advertencia recibe Tomás.** Un aviso fijo arriba (“no recupera, reconstruye”), un informe en lenguaje simple después de cada ampliación y, si SigLIP detecta texto o números, una alerta explícita: *“Compará cada número con el dashboard original antes de publicarlo”*. La imagen descargada lleva una etiqueta visible **“Ampliada con IA”** y un metadato PNG con el modelo y los parámetros usados.

**Un ejemplo concreto donde el modelo inventó.** [COMPLETAR con tu prueba real, por ejemplo: “En `texto_chico.png`, con intensidad 0.3, el modelo transformó ‘87%’ en ‘B7%’ y ‘2,3’ en ‘2,8’. El texto se ve nítido y creíble, pero es falso. La prueba de consistencia lo dio como ‘bastante consistente’: al achicarse, un 8 y un 3 borrosos se parecen. Esto muestra que consistente ≠ verdadero.” + captura]

**Cómo se puede verificar.** Solo con evidencia externa: volver a la fuente (el dashboard o el archivo original) o sacar una nueva captura en alta resolución. Las herramientas de la app indican **dónde** mirar con más atención, pero no prueban que algo sea real. Una imagen ampliada con IA llega, como mucho, a **creíble**; nunca a **verificable** por sí sola.

## Características

- Ampliación ×4 con `stabilityai/stable-diffusion-x4-upscaler`
- Control **“Cuánto puede inventar el modelo”** (mapea al parámetro `noise_level`)
- Presets por tipo de imagen (dashboard, gráfico, foto): sin prompts técnicos
- Comparador antes/después con deslizador (bicúbica vs IA)
- Mapa de detalle inventado + prueba de consistencia
- Análisis con SigLIP 2: nitidez relativa y detección de texto, números o caras
- Laboratorio de verificación con PSNR y SSIM
- Exportación con etiqueta visible y metadatos “Ampliada con IA”
- Funciona en CPU (lento, con imágenes chicas) y en GPU (Colab)

## Tecnologías Utilizadas

**Entorno:** Python 3.12, uv

**Interfaz:** Gradio [COMPLETAR versión: `uv pip show gradio`]

**Modelos de IA:**
- `stabilityai/stable-diffusion-x4-upscaler` (licencia CreativeML Open RAIL++-M): ampliación por difusión latente guiada por texto
- `google/siglip2-base-patch16-224` (licencia Apache 2.0): clasificación zero-shot de nitidez y contenido

**Procesamiento:** Pillow, NumPy, scikit-image, PyTorch, diffusers, transformers

**Hardware con el que se probó:**
- Google Colab, GPU NVIDIA T4 (16 GB VRAM): [COMPLETAR: segundos por imagen]
- Notebook propia sin GPU: [COMPLETAR: procesador, RAM, sistema operativo, minutos por imagen]

## Arquitectura del Sistema

```
Tomás ──> Interfaz Gradio (app.py)
            │  imagen + tipo + "cuánto puede inventar"
            ▼
     src/diffusion.py ── preparar (RGB, tamaño, múltiplo de 8)
            │            └─> SD x4 Upscaler (difusión) ──> imagen ×4
            ▼
     src/honesty.py ──── bicúbica (línea base sin IA)
            │            ├─> mapa "dónde inventó" (IA − bicúbica)
            │            ├─> prueba de consistencia (achicar IA vs original)
            │            └─> etiqueta + metadato "Ampliada con IA"
            ▼
     src/analysis.py ─── SigLIP 2: ¿más nítida? ¿hay texto, números, caras?
            ▼
     src/informe.py ──── informe en lenguaje simple + advertencias
            ▼
     Comparador · mapas · informe · descarga

     Pestaña Laboratorio: imagen nítida ─> degradar ×4 ─> bicúbica / IA
                          ─> src/metrics.py (PSNR, SSIM vs. la verdad)
```

| Archivo | Responsabilidad |
|---|---|
| `app.py` | Solo la interfaz y el manejo de errores para la persona usuaria |
| `src/diffusion.py` | Carga única del modelo, preparación de la imagen y ampliación |
| `src/honesty.py` | Procesamiento clásico: bicúbica, mapas de diferencias, etiqueta |
| `src/analysis.py` | SigLIP 2 zero-shot |
| `src/metrics.py` | Degradación controlada y PSNR/SSIM |
| `src/informe.py` | Texto del informe |
| `tests/` | Pruebas sin descargar modelos |

## Instalación y Ejecución

### Requisitos
- Python 3.12 y [uv](https://docs.astral.sh/uv/)
- Sin GPU: funciona en CPU, con imágenes de hasta 128 px de entrada (tarda varios minutos)
- Con GPU NVIDIA (6 GB+ de VRAM) o en Colab: hasta 512 px de entrada
- ~3 GB libres para los modelos (se descargan solos la primera vez)
- No hace falta token de Hugging Face: los dos modelos son públicos

### Pasos

```bash
git clone https://github.com/[COMPLETAR_USUARIO]/ampliador-honesto.git
cd ampliador-honesto
uv sync
uv run app.py
```

Abrí http://127.0.0.1:7860. Para correr las pruebas: `uv run pytest`.

> **PyTorch:** en Windows y Mac, `uv sync` instala la versión para CPU. Si tenés una placa NVIDIA y querés usarla, seguí la [guía de uv para PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).

### Alternativa: Google Colab (recomendada si no tenés GPU)

1. Abrí [`colab/ejecutar_en_colab.ipynb`](colab/ejecutar_en_colab.ipynb) en Colab (*Archivo → Abrir notebook → GitHub*).
2. Activá la GPU: *Entorno de ejecución → Cambiar tipo de entorno → T4 GPU*.
3. Ejecutá las celdas en orden y abrí el link `https://....gradio.live` que aparece al final.

## Uso

1. Abrí la app.
2. Subí una captura o elegí un ejemplo.
3. Elegí el tipo de imagen y cuánto puede inventar el modelo (empezá en 0.1).
4. Hacé clic en **Ampliar ×4**.
5. Arrastrá el comparador, mirá el mapa de lo inventado y leé el informe.
6. Descargá la imagen (sale con la etiqueta “Ampliada con IA”).
7. Opcional: en **Laboratorio**, subí una captura nítida para ver cuánto se acerca la IA a la verdad.

## Decisiones de Diseño

### Por qué Gradio
Tiene un comparador con deslizador (`gr.ImageSlider`) listo para usar, barra de progreso y un link público en Colab con `share=True`. Tomás no tiene que instalar nada si usa Colab.

### Por qué Stable Diffusion x4 Upscaler
Es un modelo de difusión hecho específicamente para super-resolución, público (no requiere token), el más liviano de los sugeridos y el único que podía funcionar en CPU con imágenes chicas. Su parámetro `noise_level` permite darle a Tomás un control directo sobre **cuánto puede inventar** el modelo.

### Por qué SigLIP 2
El riesgo principal para Tomás son los **números y las letras inventadas**. SigLIP permite detectar zero-shot si la imagen tiene texto, números o caras para mostrar una advertencia específica, y comparar la nitidez percibida de la versión bicúbica y la de IA. Se eligió en lugar de un VLM porque es mucho más liviano y alucina menos (no genera texto libre).

### Principios de Human-AI Interaction aplicados

**Transparencia:** aviso fijo de que la IA reconstruye; la barra de progreso explica cada etapa (“dibujando lo que falta”); se indica si corre en CPU o GPU y cuánto puede tardar.
**Control:** deslizador “cuánto puede inventar”, presets por tipo de imagen, opción de activar el segundo modelo, comparador para ir y volver.
**Explicabilidad:** informe en lenguaje simple con porcentaje de imagen inventada, veredicto de consistencia y opinión de SigLIP (siempre aclarando que puede equivocarse).
**Honestidad sobre lo generado:** mapa de lo inventado, laboratorio con evidencia, etiqueta visible y metadato en la descarga. Nunca se usa la palabra “recuperar”.
**Manejo de errores:** mensajes humanos con sugerencia (“La imagen pesa 15 MB. Probá con una de menos de 10 MB”), reintento automático con la mitad del tamaño si falta memoria, aviso cuando la imagen se achica.

## Conceptos de Procesamiento Digital Aplicados

- **Muestreo e interpolación:** bicúbica como línea base (`honesty.ampliar_bicubica`); LANCZOS para achicar con anti-aliasing (`diffusion.preparar_imagen`); vecino más cercano para mostrar la versión degradada “pixelada”.
- **Resolución espacial y aliasing:** reducción por promedio de bloques (`BOX`) para simular baja resolución (`metrics.degradar`, `honesty.prueba_consistencia`).
- **Espacios de color:** conversión RGBA→RGB con composición sobre blanco; cálculo de luminancia Y (ITU-R 601) para comparar detalle (`honesty.luminancia`).
- **Filtrado espacial:** filtro gaussiano (pasa-bajos) para convertir diferencias de píxeles sueltos en zonas legibles (`honesty.mapa_detalle_agregado`).
- **Operaciones aritméticas entre imágenes:** diferencia absoluta para los mapas; normalización y umbralización para calcular el porcentaje inventado.
- **Restauración y modelo de degradación:** degradamos de forma controlada (achicar ×4) para tener referencia y medir.
- **Métricas de calidad:** PSNR y SSIM contra la imagen de referencia (`src/metrics.py`).
- **Pseudocolor:** mapa de calor negro→rojo→amarillo superpuesto sobre la imagen en gris (`honesty.colorear_mapa`).

## Uso de Agentes de IA

- **Herramientas utilizadas:** Claude (Anthropic) como asistente de código y documentación.
- **Qué decidí yo:** [COMPLETAR con tus decisiones reales. Ejemplos: la sub-tarea A y la persona (creador de contenido de BI, porque es mi mundo); usar Colab porque no tengo GPU; priorizar el riesgo de los números inventados; qué pruebas hacer y qué ejemplos usar para la falla del video.]
- **Qué propuso el agente y cómo lo revisé:** la estructura de módulos, la prueba de consistencia (achicar el resultado y compararlo con el original), el laboratorio con PSNR/SSIM y el código base de cada archivo. Leí cada archivo, lo ejecuté con mis imágenes y corrí las pruebas de `tests/`. [COMPLETAR: qué ajustaste vos]
- **Qué descarté y por qué:** [COMPLETAR. Ej.: agregar un VLM (SmolVLM) para describir cambios: lo descarté por tiempo y memoria, y porque puede alucinar justo sobre los números.]

## Limitaciones Conocidas

- Funciona mejor con gráficos y dashboards simples; con **texto chico inventa letras y números** (es el caso de riesgo principal).
- Tiempo de procesamiento: GPU T4 ≈ [COMPLETAR] s; CPU ≈ [COMPLETAR] min por imagen de 128 px.
- Tamaño máximo de entrada: 512 px en GPU y 128 px en CPU (la salida es ×4). Archivos de hasta 10 MB.
- La prueba de consistencia **no detecta** inventos que, al achicarse, se ven igual que el original.
- SigLIP no fue entrenado para medir calidad y puede clasificar mal; sus porcentajes son relativos entre las frases que le damos.
- Los prompts del modelo están en inglés; el modelo puede sesgar el estilo hacia “dashboard genérico”.

## Métricas de Calidad (opcional)

- **PSNR (dB):** parecido píxel a píxel con la imagen real. Más alto = más parecido.
- **SSIM (0 a 1):** parecido de estructura (bordes, contraste). Más alto = más parecido.
- **Para Tomás:** si la IA *se ve* más nítida que la bicúbica pero tiene PSNR/SSIM más bajos, esa nitidez es inventada. [COMPLETAR con tus números del laboratorio para los 3 ejemplos]
- Solo se pueden calcular en el Laboratorio, donde existe la imagen real. Miden parecido, no autenticidad.

## Trabajo Futuro

- [ ] Detectar zonas de texto (OCR) y marcarlas automáticamente como “no confiables”
- [ ] Comparar con otro modelo de difusión (ControlNet tile)
- [ ] Explicación de cambios con un VLM local
- [ ] Publicarlo como Hugging Face Space

## Reflexiones y Aprendizajes

[COMPLETAR con 1-2 párrafos tuyos. Ideas: qué te sorprendió cuando el modelo inventó un número; qué aprendiste del trabajo con el agente (leer el diff, pedir pruebas); cómo cambió tu forma de mirar una imagen “mejorada”.]

## Recursos y Referencias

- Modelo de difusión: https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler
- SigLIP 2: https://huggingface.co/google/siglip2-base-patch16-224
- Documentación de diffusers: https://huggingface.co/docs/diffusers/
- Clasificación zero-shot en transformers: https://huggingface.co/docs/transformers/tasks/zero_shot_image_classification
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*: https://arxiv.org/abs/2112.10752
- Simon Willison, *Vibe engineering*: https://simonwillison.net/2025/Oct/7/vibe-engineering/
- Gradio: https://www.gradio.app/docs · uv: https://docs.astral.sh/uv/

## Autor

**Gabriel Márquez**  
Estudiante de Tecnicatura Superior en Ciencias de Datos e IA - IFTS 24  
Materia: Procesamiento Digital de Imágenes e Introducción a Visión por Computadora  
Profesor: Matías Barreto · Año: 2026

[COMPLETAR: links a GitHub, LinkedIn, portfolio]

## Licencia

Código bajo licencia MIT (ver `LICENSE`). Los modelos mantienen sus propias licencias: CreativeML Open RAIL++-M (SD x4 Upscaler) y Apache 2.0 (SigLIP 2). Las imágenes de `assets/ejemplos/` son sintéticas, creadas para este proyecto, con datos ficticios.

---

**Trabajo Integrador N°2 - IFTS 24 - 2026**
