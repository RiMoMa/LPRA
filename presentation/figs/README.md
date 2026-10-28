# Figuras de la presentación

Todos los scripts se ejecutan desde la raíz del repositorio y leen `results/` o
`results.csv`; ninguna cifra se copia a mano. Para regenerar todo:

```bash
.venv/bin/python presentation/figs/make_all.py
```

| Figura | Script | Fuente |
|---|---|---|
| `pipeline.svg` | `make_pipeline.py` | tabla «Pipeline» de `README.md` |
| `yolov8n_arch.svg` | `make_arch.py` | `LEARNING_NOTES.md`; `YOLO(...).info()` |
| `dfl.svg` | `make_dfl.py` | `LEARNING_NOTES.md`, «Graph structure» |
| `quant_affine.png` | `make_quant_affine.py` | fórmula y conceptos de `LEARNING_NOTES.md` |
| `latency_vs_map_phase2.png` | `make_latency_vs_map.py` | `results.csv`, fase 2 |
| `latency_vs_map_phase2_zoom.png` | `make_latency_vs_map.py` | `results.csv`, fase 2; eje y ampliado, sin los modelos con mAP 0 |
| `checks.png` | `make_checks.py` | `results.csv`, fase 2-check |
| `split.png` | `make_split.py` | `results.csv`, fase 3-split |
| `split_graph.svg` | `make_split_graph.py` | modelos ONNX; escalas `QuantizeLinear` |
| `qualitative_large.png`, `qualitative_small.png` | `make_qualitative.py` | `results/dets/*.json`, `data/split.json` |
| `qualitative_both.png` | `make_qualitative.py` | composición de las dos filas anteriores |

`results/activation_histogram.png` ya existe y se reutiliza; `make_all.py`
verifica que esté presente.

Las figuras que muestran latencia indican la corrida en el título o en el bloque
de evidencia de la diapositiva. No se debe comparar una latencia de `checks.png`
con una de `split.png`.
