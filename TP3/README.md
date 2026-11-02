# TP3 — Modelado del error de predicción (rama AvancesTP3Matias)

Trabajo en curso de Matías Bacalhau. Consigna: `Consigna TP3.pdf`. Las decisiones tomadas y su
justificación están en `DECISIONES.md` (se actualiza a medida que avanza el trabajo).

## Estado
Hecho: preparación de datos, validación walk-forward, baselines (B0–B3) y feature engineering (familias 1–4).
Pendiente: modelo interpretable, modelo de mínimo error, interpretabilidad (Etapa B), dashboard (Etapa C) e informe.

## Cómo reproducir
Los datos no están en el repo (pesan ~700 MB). Hacen falta los CSV diarios del TP1
(`forecast_observation_2026_MM_DD.csv`); en `config.yaml`, `entrada` es la carpeta donde están.

```bash
cd TP3
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# en macOS, LightGBM necesita OpenMP:  brew install libomp
.venv/bin/python src/preparar_datos.py     # incremental: solo procesa los días nuevos
.venv/bin/python src/validacion.py         # muestra los pliegues walk-forward
.venv/bin/python src/baselines.py          # baselines B0-B3 en los pliegues
.venv/bin/python src/features.py           # variables (familias 1-4) por día
.venv/bin/python src/verificar_features.py # controles de causalidad y recálculo independiente
```

## Archivos
- `src/preparar_datos.py` — CSV diarios → parquet por día con el target `error_c = observado − pronosticado`.
- `src/validacion.py` — walk-forward con ventana expansiva y gap, cortes por día de la hora objetivo.
- `src/baselines.py` — NWS crudo, sesgo global, sesgo por estación × hora solar, persistencia.
- `src/features.py`, `src/verificar_features.py` — variables conocidas al emitir el pronóstico y su control.
- `config.yaml` — parámetros (rutas, hueco máximo de interpolación, validación, α del sesgo exponencial).
