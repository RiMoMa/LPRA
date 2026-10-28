# Brief para generar la presentación del proyecto LPRA

Este archivo es la instrucción completa para el agente que construye la presentación (opencode con DeepSeek u
otro). Léalo entero antes de escribir una sola diapositiva. Todo lo que necesita está en este repositorio.

## 0. Parámetros (cámbielos aquí si hace falta)

- **Idioma:** español técnico. Los términos que en la industria se dicen en inglés se dejan en inglés, en cursiva,
  con su traducción la primera vez (ej.: *per-channel*, por canal).
- **Audiencia:** comité técnico de una entrevista de trabajo en ML / visión por computador / despliegue de modelos.
  Saben de redes neuronales; no necesariamente de ONNX ni de cuantización.
- **Duración objetivo:** 20 a 25 minutos, 22 a 28 diapositivas, más un apéndice.
- **Formato:** Markdown para Marp, en `presentation/slides.md`, exportable a PDF y PPTX:
  `npx @marp-team/marp-cli presentation/slides.md --pdf` y `--pptx`.
- **Figuras:** generadas por scripts en `presentation/figs/make_*.py` que leen los archivos de `results/`.
  Ninguna figura se dibuja a mano con números copiados.

## 1. Fuentes de verdad, en este orden

1. `results.csv` (tabla larga generada por `10_collect_results.py`) y los archivos crudos de `results/`.
2. `LEARNING_NOTES.md`: razonamiento, hipótesis, predicciones, errores cometidos.
3. `INTERVIEW_NOTES.md`: el hilo narrativo y los mensajes clave.
4. Los scripts numerados `00_…` a `13_…`: qué se hizo exactamente y con qué parámetros.
5. `README.md`: tablas resumen.

Si dos fuentes no coinciden, manda la 1, y la diapositiva lo dice.

## 2. Reglas que no se negocian

1. **Ningún número sin fuente.** Cada cifra de una diapositiva sale de `results.csv`, de un archivo de `results/`
   o de `LEARNING_NOTES.md`. En las notas del orador de esa diapositiva se escribe la fuente
   (archivo y fila o sección). Si una cifra no está en el repositorio, no se usa.
2. **No comparar latencias entre corridas distintas.** Hay tres corridas de benchmark, cada una con su propia
   fila FP32: fase 2 (`latency_raw.csv`), chequeos (`latency_checks.csv`, 2026-10-25) y fase 3
   (`latency_split.csv`, 2026-10-28). Una tabla o gráfica de latencia usa una sola corrida y lo dice en el título.
3. **Toda abreviatura se define antes de usarse**, en la misma diapositiva donde aparece por primera vez,
   además del glosario de la sección 4. Ejemplo: "ONNX (*Open Neural Network Exchange*), un formato abierto de grafo…".
   Una sigla que aparezca sin definir antes es un error que hay que corregir.
4. **Resultados pendientes se marcan como pendientes.** Si `results/attribution_split.json` no existe, las
   diapositivas del experimento de atribución muestran la hipótesis y la predicción, con la etiqueta
   "Resultado pendiente". Nunca se rellena con un valor plausible.
5. **Hipótesis y hechos se distinguen visualmente.** Hecho medido: sin marca. Hipótesis no verificada: etiqueta
   "Hipótesis". Predicción que falló: etiqueta "Predicción fallida" y qué se aprendió. Las fallas se muestran,
   no se esconden: son parte del valor del proyecto.
6. **Cada decisión técnica se presenta con la misma estructura de cuatro partes:**
   Pregunta → Hipótesis o razón → Decisión → Evidencia (cifra, gráfica o grafo).
7. **Datos de la arquitectura que no estén en el repositorio se obtienen ejecutando código**, no de memoria.
   Ejemplo: número de parámetros y GFLOPs con
   `python -c "from ultralytics import YOLO; YOLO('weights/yolov8n.pt').info()"`, y se cita esa salida.

## 3. Hilo narrativo

Una frase que la presentación entera debe sostener:

> Cuantizar a INT8 dio 1,41× de velocidad con −0,33 puntos de mAP@0.5, pero cada problema serio que apareció
> (cero detecciones, un modelo INT8 más lento que FP32, la pérdida del modelo dividido) solo se vio al
> inspeccionar el grafo que ONNX Runtime ejecuta, no el archivo ni el código.

## 4. Glosario (diapositiva 3, y cada término se define otra vez en su primer uso)

Usar estas definiciones; están revisadas contra el proyecto.

| Sigla o término | Significado | En este proyecto |
|---|---|---|
| ML | *Machine Learning*, aprendizaje automático | — |
| CNN | *Convolutional Neural Network*, red neuronal convolucional | YOLOv8n es una CNN |
| YOLO | *You Only Look Once*: detector de una sola pasada que predice cajas y clases en un único *forward* | YOLOv8n, la variante *nano* de Ultralytics |
| ALPR | *Automatic License Plate Recognition*, reconocimiento automático de placas | Caso de uso objetivo; aquí se evalúa la etapa previa: detectar vehículos |
| COCO | *Common Objects in Context*, conjunto de datos con 80 clases | val2017, clases car, motorcycle, bus, truck |
| ONNX | *Open Neural Network Exchange*, formato abierto que describe un modelo como grafo de operadores | El modelo exportado `.onnx` |
| ORT | ONNX Runtime, el motor que ejecuta grafos ONNX | Versión 1.31 |
| EP | *Execution Provider*, el backend de hardware que usa ORT | `CPUExecutionProvider` |
| Opset | Versión del conjunto de operadores ONNX que el archivo puede usar | 17 |
| FP32 | Punto flotante de 32 bits | Modelo de referencia |
| INT8 / U8 / S8 | Entero de 8 bits; U8 sin signo (0 a 255), S8 con signo (−128 a 127) | Activaciones U8, pesos S8 |
| PTQ | *Post-Training Quantization*, cuantizar un modelo ya entrenado, sin reentrenar | Método usado |
| QAT | *Quantization-Aware Training*, entrenar simulando la cuantización | No se usó; mencionar como alternativa |
| Q / DQ | Nodos `QuantizeLinear` / `DequantizeLinear` de ONNX | Marcan qué tensores son INT8 |
| QDQ | Formato de cuantización que anota el grafo FP32 con pares Q→DQ | Formato usado |
| QOperator | Formato alternativo con operadores enteros explícitos (`QLinearConv`, …) | Mencionar como contraste |
| *Scale*, *zero point* | Parámetros de la cuantización afín: q = round(x/s) + z, x̂ = s·(q − z) | Calculados por calibración |
| MinMax | Calibración que fija el rango con el mínimo y el máximo observados | Método usado |
| *Per-tensor* / *per-channel* | Una escala por tensor / una escala por canal de salida de los pesos | Diseño 2×2 de la fase 2 |
| BN | *Batch Normalization*; en inferencia se pliega (*folding*) dentro de los pesos de la convolución | Explica por qué hace falta *per-channel* |
| SiLU | *Sigmoid Linear Unit*, x·σ(x), la activación de YOLOv8 | En ONNX aparece como `Sigmoid` → `Mul` |
| C2f | Bloque de YOLOv8 derivado de CSP con dos convoluciones y concatenación de ramas | Backbone y neck |
| CSP | *Cross Stage Partial*, dividir los canales en dos ramas y unirlas | Origen de C2f |
| SPPF | *Spatial Pyramid Pooling – Fast*, tres MaxPool en serie | 3 nodos MaxPool |
| FPN / PAN | *Feature Pyramid Network* / *Path Aggregation Network*: combinan mapas de varias escalas, de arriba abajo y de abajo arriba | El neck; 2 `Resize` |
| DFL | *Distribution Focal Loss*: cada lado de la caja se predice como una distribución sobre 16 posiciones y se toma su valor esperado | Reshape → Softmax → Conv 1×1 con pesos 0…15 |
| Stride | Cuántos píxeles de la imagen representa una celda del mapa | 8, 16 y 32 |
| NMS | *Non-Maximum Suppression*, eliminar cajas repetidas | Fuera del grafo, en Python |
| IoU | *Intersection over Union* entre dos cajas | Umbral de acierto en mAP |
| AP / mAP | *Average Precision* por clase / su media entre clases | mAP@0.5 y mAP@.5:.95 |
| mAP@0.5 | mAP con IoU ≥ 0,5 | Métrica principal |
| mAP@.5:.95 | mAP promediado en IoU de 0,50 a 0,95 en pasos de 0,05; exige cajas más precisas | Métrica de localización |
| FLOPs | Operaciones de punto flotante | Para decir que la cabeza es ~0 % del cómputo |
| SIMD | *Single Instruction, Multiple Data* | Base de AVX |
| AVX2 / AVX-512 | Extensiones SIMD de x86 de 256 y 512 bits | La CPU tiene ambas |
| VNNI | *Vector Neural Network Instructions*; `vpdpbusd` multiplica u8 × s8 y acumula en int32 | Razón de U8×S8 y de no usar *reduce_range* |
| CCD | *Core Complex Die*, cada chiplet de núcleos del Ryzen | Benchmark fijado en CCD0 (CPUs 0–5) |
| L3 / V-Cache | Caché de nivel 3 / caché 3D apilada de AMD | CCD0 tiene 96 MB |
| p95 | Percentil 95 de la latencia | Junto a la media |
| ATen | Biblioteca de operadores tensoriales de PyTorch | El exportador traduce ATen → ONNX |
| NCHW / NCHWc | Distribución de memoria lote-canal-alto-ancho / variante con canales en bloques | ORT usa NCHWc para convoluciones AVX-512 |

## 5. Estructura propuesta, diapositiva por diapositiva

### Bloque A · Contexto (4)
1. **Título.** Exportación ONNX y cuantización INT8 de YOLOv8n en CPU. Nombre, fecha.
2. **El problema.** ALPR necesita detectar vehículos rápido en CPU. Pregunta del proyecto: ¿cuánta velocidad da
   INT8 y cuánta precisión cuesta? Qué *no* se hizo: no hay reentrenamiento, no hay GPU, no hay lectura de placas.
3. **Glosario** (sección 4, versión compacta; el resto en apéndice).
4. **Método en una imagen.** Diagrama del pipeline numerado 00 → 13 con la salida de cada script.
   Fuente: tabla "Pipeline" de `README.md`.

### Bloque B · La red por dentro (5)
5. **YOLOv8n a vista de pájaro.** Entrada `[B,3,640,640]` → backbone (Conv + C2f + SPPF) → neck (FPN/PAN) →
   cabeza desacoplada sin anclas a tres escalas. Parámetros y GFLOPs: obtenerlos con `info()` (regla 7).
6. **Bloques básicos.** Conv + BN + SiLU; C2f; SPPF. Mostrar cómo SiLU aparece en ONNX como `Sigmoid` → `Mul`
   (57 SiLU + 1 Sigmoid de clases = 58 Sigmoid). Fuente: `LEARNING_NOTES.md`, "Graph structure".
7. **La cabeza y las 8400 predicciones.** Tres mapas con stride 8, 16 y 32: 80×80 + 40×40 + 20×20 = 8400 puntos.
   Rama `cv2` (cajas, 4×16 valores DFL) y rama `cv3` (80 clases). No hay *objectness* en YOLOv8.
8. **DFL explicado.** Cada lado de la caja es una distribución sobre 16 posiciones: Softmax y valor esperado
   con una Conv 1×1 de pesos fijos 0…15. Luego la decodificación de anclas (Sub/Add/Div) y × stride a píxeles.
9. **La salida `output0 [B,84,8400]`.** 4 coordenadas de caja (centro x, centro y, ancho, alto) en píxeles
   + 80 probabilidades. Remarcar ya aquí: **dos magnitudes muy distintas en un solo tensor**. Es la semilla de
   la fase 2. NMS va fuera del grafo.

### Bloque C · Fase 1, exportación (3)
10. **Qué es exportar.** Trazado (*tracing*) con `torch.onnx.export`: se ejecuta una pasada y cada operador ATen
    se traduce a ONNX a un opset fijo; los pesos pasan a *initializers*. Legacy frente a dynamo.
11. **Decisiones de exportación** (estructura de cuatro partes, regla 6): `fuse()` antes de exportar (BN plegada),
    ejes dinámicos de lote (363 → 407 nodos y por qué), opset 17 (necesario para QDQ *per-channel*).
12. **Paridad numérica.** Cajas máx. |Δ| = 8,9e-3 px y scores 5,3e-5 en 50 imágenes; la detección principal
    coincide en todas. Por qué no es cero: suma en punto flotante no asociativa. Por qué importa: sin paridad
    no se puede atribuir ninguna pérdida posterior a la cuantización. Fuente: `results/parity_coco50.json`.

### Bloque D · Fase 2, cuantización (7)
13. **Cuantización afín en una diapositiva.** Fórmula, *scale* y *zero point*, error de recorte frente a error
    de redondeo. Una gráfica pequeña de x frente a x̂.
14. **Calibración estática y separación de datos.** 150 imágenes de calibración, 646 de evaluación, semilla 0,
    disjuntas. Por qué: calibrar con imágenes de evaluación sería fuga de datos. Fuente: `04_split.py`.
15. **QDQ: el archivo no es lo que se ejecuta.** Q→DQ anotan; ORT fusiona `DQ→Op→Q` en kernels enteros al
    cargar. Mostrar conteo de nodos del grafo en ejecución (`QLinearConv` ×64, etc.). Fuente: `06_inspect_qdq.py`.
16. **Decisiones de configuración.** U8 para activaciones y S8 para pesos (operandos nativos de VNNI);
    *per-channel* para pesos (BN plegada deja canales con rangos 16× distintos); `reduce_range=False` (VNNI
    acumula en int32, no hay saturación de int16).
17. **El peligro de `output0`.** Histograma `results/activation_histogram.png`. Escala 2,618, zero point 4:
    todo score redondea a 0 → 0 detecciones, mAP 0. Se predijo antes de medir.
18. **Diseño 2×2 y resultados.** *per-tensor*/*per-channel* × todo cuantizado/cabeza FP32. Tabla de latencia y
    mAP de la fase 2 (una sola corrida, `latency_raw.csv` + `map_overall.json`). Gráfica latencia vs mAP@0.5.
19. **Perfilado: por qué 1,4× y no 3×.** Conv INT8 3,0× más rápida a nivel de kernel; se diluye por SiLU
    (QLinearSigmoid + QLinearMul) y QLinearConcat. Un solo nodo FP32 (Softmax del DFL) cuesta 3,25 ms.
    Fuente: `LEARNING_NOTES.md`, "Profiling".

### Bloque E · Chequeos de un solo ajuste (2)
20. **`reduce_range=True`:** latencia igual; efecto en precisión no establecido (las dos métricas no coinciden
    en signo y no hay intervalos de confianza). Decir eso tal cual.
21. **Activaciones S8, el valor por defecto de ORT: 32,45 ms, 0,50×, más lento que FP32.** Predicción fallida
    ("igual o algo más lento"). Solo 6 de 64 convoluciones se fusionan. Lección: mirar el grafo en ejecución.
    Fuente: `latency_checks.csv`, `map_checks.json`.

### Bloque F · Fase 3, cirugía del grafo (4)
22. **Arreglar el grafo en vez de excluir nodos.** Borrar `Concat_9`, exponer `boxes` y `scores`. Paridad
    exacta (máx. |Δ| = 0). Diagrama antes/después. Fuente: `12_split_outputs.py`.
23. **Resultados de la división** (corrida 2026-10-28, `latency_split.csv` + `map_split.json`): 8,88 ms, 1,88×,
    mAP@0.5 0,5509. Domina a `pc_all`, no a `pc_headfp32`.
24. **Predicción fallida en magnitud.** El paso de ±1,3 px de la salida no explica 5,3 puntos con IoU 0,5.
    Tabla de escalas intermedias: `Concat_8` en unidades de grilla, paso 0,314 celdas = hasta 10 px a stride 32;
    piso de scores en 0,0038. AP por clase: car −11,5, bus −1,3. Fuente: `LEARNING_NOTES.md`, "Attribution".
25. **Atribución.** Variantes `_boxq` y `_scoreq`. Si `results/attribution_split.json` existe: tabla con mAP
    y AP por tamaño. Si no: predicción + "Resultado pendiente" (regla 4).

### Bloque G · Cierre (3)
26. **Resultados cualitativos.** Una misma imagen de evaluación con las cajas de FP32, `pc_headfp32` y
    `split_pc_all` (umbral de visualización 0,25), dibujadas desde `results/dets/*.json`. Elegir 2 imágenes:
    una con vehículos grandes y otra con vehículos pequeños. Script: `presentation/figs/make_qualitative.py`.
27. **Qué aprendí** (tres a cinco viñetas, de `INTERVIEW_NOTES.md`) y **qué haría después**: intervalos de
    confianza con *bootstrap* pareado, calibración por percentil o entropía, QAT, prueba con imágenes ALPR reales.
28. **Recomendación final.** `yolov8n_int8_pc_headfp32.onnx`: 1,41×, −0,33 mAP@0.5, y por qué no los otros.

### Apéndice
- Glosario completo; tablas completas de `results.csv` por fase; entorno (`results/env.json`); comandos para
  reproducir; lista de errores cometidos ("Mistakes made" de `LEARNING_NOTES.md`).

## 6. Figuras a generar (todas desde archivos, con script)

| Archivo | Contenido | Datos |
|---|---|---|
| `figs/pipeline.svg` | Pipeline 00 → 13 | Tabla de `README.md` |
| `figs/yolov8n_arch.svg` | Backbone, neck, cabeza con formas de tensores | Sección 5, diapositivas 5–9 |
| `figs/dfl.svg` | Softmax de 16 posiciones → valor esperado → × stride | `LEARNING_NOTES.md` |
| `figs/quant_affine.png` | x frente a x̂ con recorte y redondeo | Fórmula de la sección 4 |
| `results/activation_histogram.png` | Ya existe; reutilizar | — |
| `figs/latency_vs_map_phase2.png` | Dispersión latencia vs mAP@0.5, fase 2 | `results.csv`, fase 2 |
| `figs/checks.png` | Barras de latencia de los chequeos | `results.csv`, fase 2-check |
| `figs/split.png` | Barras de latencia y mAP, fase 3 | `results.csv`, fase 3-split |
| `figs/split_graph.svg` | `Concat_9` antes/después | `12_split_outputs.py` |
| `figs/qualitative_*.png` | Cajas de tres modelos en la misma imagen | `results/dets/*.json` + `data/coco` |

Para el diagrama de la red puede abrir el modelo en Netron (`netron models/yolov8n.onnx`) y tomar capturas
recortadas de la cabeza, pero cada captura debe señalar algo concreto (por ejemplo, `Concat_9`).

## 7. Verificación antes de entregar

- [ ] Cada sigla está definida antes de su primer uso (buscar en el texto todas las palabras en mayúsculas).
- [ ] Cada número tiene su fuente en las notas del orador.
- [ ] Ninguna tabla mezcla latencias de corridas distintas.
- [ ] Hipótesis, hechos y predicciones fallidas están marcados como tales.
- [ ] Nada de la fase de atribución se presenta como resultado si `attribution_split.json` no existe.
- [ ] Los scripts de `presentation/figs/` se ejecutan desde la raíz del repositorio y regeneran todas las figuras.
- [ ] El PDF exportado se revisó página por página: sin texto cortado ni figuras ilegibles.
