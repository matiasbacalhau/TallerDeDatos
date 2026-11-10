# TP3 — Modelado del error de predicción (rama AvancesTP3Matias)

Trabajo en curso de Matías Bacalhau. Consigna: `Consigna TP3.pdf`. Las decisiones tomadas y su
justificación están en `DECISIONES.md` (se actualiza a medida que avanza el trabajo).

## Estado
Hecho: preparación de datos, control de calidad con exclusión automática de días y alarmas, validación walk-forward, baselines (B0–B3) y feature engineering (familias 1–4).
Hecho también: modelo interpretable (regresión lineal Ridge con curvas, variables por criterio meteorológico).
Pendiente: modelo de mínimo error, interpretabilidad (Etapa B), dashboard (Etapa C) e informe.

## Cómo reproducir
Los datos no están en el repo (pesan ~700 MB). Hacen falta los CSV diarios del TP1
(`forecast_observation_2026_MM_DD.csv`); en `config.yaml`, `entrada` es la carpeta donde están.

```bash
cd TP3
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# en macOS, LightGBM necesita OpenMP:  brew install libomp
.venv/bin/python src/pipeline.py           # todo en orden (correr cada vez que llegan días nuevos)
# o paso a paso:
.venv/bin/python src/preparar_datos.py     # incremental: solo procesa los días nuevos
.venv/bin/python src/calidad_dias.py       # excluye días con faltantes anómalos y deja el registro
.venv/bin/python src/validacion.py         # muestra los pliegues walk-forward
.venv/bin/python src/baselines.py          # baselines B0-B3 en los pliegues
.venv/bin/python src/features.py           # variables por día (familias 1-4; la 5, vecinos, es experimental)
.venv/bin/python src/verificar_features.py # controles de causalidad y recálculo independiente
.venv/bin/python src/modelo_interpretable.py # elige el α del sesgo (pliegues 1-2) y reporta (3-4)
.venv/bin/python src/modelo_lgbm.py         # elige grilla y árboles (pliegues 1-2) y reporta (3-4)
.venv/bin/python src/comparar_modelos.py    # ¿las diferencias entre modelos superan el ruido?
.venv/bin/python src/interpretar.py         # Etapa B: curvas, SHAP, sesgo restante y techo (figuras/)
```

## Archivos
- `src/preparar_datos.py` — CSV diarios → parquet por día con el target `error_c = observado − pronosticado`.
- `src/calidad_dias.py` — regla de exclusión de días, marca de historia incompleta y alarmas de cobertura; registro en `data/calidad/REGISTRO_CALIDAD.md`.
- `src/pipeline.py` — corre todos los pasos en orden y se detiene si un control falla.
- `src/validacion.py` — walk-forward con ventana expansiva y gap, cortes por día de la hora objetivo.
- `src/baselines.py` — NWS crudo, sesgo global, sesgo por estación × hora solar, persistencia.
- `src/features.py`, `src/verificar_features.py` — variables conocidas al emitir el pronóstico y su control.
- `src/modelado.py`, `src/modelo_interpretable.py` — piezas comunes de los modelos y la regresión lineal con curvas.
- `src/modelo_lgbm.py` — LightGBM con todas las features; compara contra B0, B2 y el modelo interpretable.
- `src/comparar_modelos.py` — prueba de significancia entre modelos con el día de prueba como unidad.
- `src/interpretar.py` — Etapa B: figuras de interpretación en `figuras/`.
- Experimento con vecinos: `--vecinos` en `modelo_interpretable.py`, `modelo_lgbm.py` y `comparar_modelos.py` (salidas `*_exp_vecinos`).
- `src/diagnostico_viento.py` — ¿la dirección del viento explica error por estación? (pliegues de elección).
- `config.yaml` — parámetros (rutas, hueco máximo de interpolación, calidad, validación, α del sesgo exponencial).
