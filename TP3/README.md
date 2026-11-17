# TP3 — Modelado del error de predicción (rama AvancesTP3Matias)

Trabajo en curso de Matías Bacalhau. Consigna: `Consigna TP3.pdf`. Las decisiones tomadas y su
justificación están en `DECISIONES.md` (se actualiza a medida que avanza el trabajo).

## Estado
Hecho: preparación de datos, control de calidad con exclusión automática de días y alarmas, validación walk-forward con pliegues anclados, baselines (B0–B3) y feature engineering (familias 1–8).
Hecho también: experimentación por familias (`notebooks/experimentos.ipynb`), Ridge de caja de cristal con las familias que elige la regla de recorte y LightGBM con las 8 familias.
**Pliegues anclados:** los bloques de prueba se cuentan desde el 30/09/2026, de a 5 días usables. Un día nuevo no cambia los pliegues anteriores; los modelos solo entrenan el pliegue que se completa (el resto sale de la caché, `data/cache/`). Los hiperparámetros se eligen una vez en los pliegues 1–2 y quedan congelados (`--reelegir` para volver a elegirlos).
Pendiente: informe y presentación.

## Ver el dashboard (Etapa C)

No hacen falta los datos crudos ni correr el pipeline: los datos del dashboard están en `app/datos/` (~26 MB).
Requiere Python 3.11 o más nuevo y git.

macOS o Linux:
```bash
git clone https://github.com/matiasbacalhau/TallerDeDatos.git
cd TallerDeDatos/TP3
python3 -m venv .venv-app && .venv-app/bin/pip install -r app/requirements.txt
.venv-app/bin/streamlit run app/dashboard.py
```

Windows (PowerShell):
```powershell
git clone https://github.com/matiasbacalhau/TallerDeDatos.git
cd TallerDeDatos\TP3
py -3 -m venv .venv-app; .venv-app\Scripts\pip install -r app/requirements.txt
.venv-app\Scripts\streamlit run app/dashboard.py
```

Se abre en el navegador en http://localhost:8501. La primera carga tarda unos segundos (6,8 millones de filas).
Para regenerar `app/datos/` con datos nuevos: correr el pipeline completo y después `src/preparar_dashboard.py`.

## Cómo reproducir
Los datos no están en el repo (pesan ~700 MB). Hacen falta los CSV diarios del TP1
(`forecast_observation_2026_MM_DD.csv`); en `config.yaml`, `entrada` es la carpeta donde están.

```bash
cd TP3
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# en macOS, LightGBM necesita OpenMP:  brew install libomp
.venv/bin/python src/pipeline.py           # datos, variables y modelo producto (correr cada vez que llegan días nuevos)
# o paso a paso:
.venv/bin/python src/preparar_datos.py     # incremental: solo procesa los días nuevos
.venv/bin/python src/calidad_dias.py       # excluye días con faltantes anómalos y deja el registro
.venv/bin/python src/validacion.py         # muestra los pliegues walk-forward
.venv/bin/python src/baselines.py          # baselines B0-B3 en los pliegues
.venv/bin/python src/features.py           # variables por día (familias 1-4; la 5, vecinos, es experimental)
.venv/bin/python src/verificar_features.py # controles de causalidad y recálculo independiente
.venv/bin/python src/producto.py           # entrena el modelo producto (config.yaml: producto.modelo) -> data/producto/
# parte experimental (no la corre el pipeline):
.venv/bin/python -m pytest -q             # tests (pliegues, caché, Ridge, experimentación)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/experimentos.ipynb   # elige las familias del Ridge
.venv/bin/python src/modelos/ridge.py      # Ridge: elige α (pliegues 1-2, una vez) y reporta (3 en adelante)
.venv/bin/python src/modelos/lgbm.py       # LightGBM: elige grilla y árboles (pliegues 1-2, una vez) y reporta
.venv/bin/python src/modelos/ols_rfe.py    # OLS + RFE: elige las variables (pliegues 1-2) y reporta
.venv/bin/python src/comparar_modelos.py    # ¿las diferencias entre modelos superan el ruido?
.venv/bin/python src/interpretar.py         # Etapa B: curvas, SHAP, sesgo restante y techo (figuras/)
.venv/bin/python src/preparar_dashboard.py  # Etapa C: datos y bandas de confianza del dashboard
.venv/bin/streamlit run app/dashboard.py    # Etapa C: abre el dashboard en http://localhost:8501
```

## Archivos
- `src/preparar_datos.py` — CSV diarios → parquet por día con el target `error_c = observado − pronosticado`.
- `src/calidad_dias.py` — regla de exclusión de días, marca de historia incompleta y alarmas de cobertura; registro en `data/calidad/REGISTRO_CALIDAD.md`.
- `src/pipeline.py` — corre todos los pasos en orden (datos → variables → modelo producto) y se detiene si un control falla.
- `src/validacion.py` — walk-forward con ventana expansiva y gap, cortes por día de la hora objetivo.
- `src/baselines.py` — NWS crudo, sesgo global, sesgo por estación × hora solar, persistencia.
- `src/features.py`, `src/verificar_features.py` — variables conocidas al emitir el pronóstico y su control.
- `src/modelado.py` — piezas comunes de los modelos (carga por pliegue, evaluación contra los baselines).
- `src/modelos/` — **parte experimental**, un archivo por tipo de modelo, todos con la misma interfaz de producto:
  - `ridge.py` — Ridge con curvas para cualquier lista de familias (caja de cristal).
  - `lgbm.py` — LightGBM con las 8 familias; compara contra B0, B2 y el Ridge.
  - `ols_rfe.py` — OLS sin penalización con variables elegidas por RFE (caja de cristal); salidas en `salidas/rfe/`.
- `src/producto.py` — **producto**: entrena con todos los días el modelo que dice `config.yaml` (`producto.modelo`) y lo guarda con su ficha en `data/producto/`.
- `src/cache.py` — caché por pliegue con huella del contenido de los datos.
- `src/ridge_gram.py` — muchos Ridge (variantes × α) desde una sola XᵀX por pliegue.
- `src/experimentos.py`, `notebooks/experimentos.ipynb` — experimentación por familias (exploración, ablación, regla de recorte); salidas en `salidas/experimentos/`.
- `src/comparar_modelos.py` — prueba de significancia entre modelos con el día de prueba como unidad.
- `src/interpretar.py` — Etapa B: figuras de interpretación en `figuras/`.
- `src/preparar_dashboard.py`, `app/dashboard.py`, `.streamlit/config.toml` — Etapa C: dashboard de confianza del pronóstico.
- `src/diagnostico_viento.py` — ¿la dirección del viento explica error por estación? (pliegues de elección).
- `config.yaml` — parámetros (rutas, hueco máximo de interpolación, calidad, validación y ancla de los pliegues, corte y umbrales de los experimentos, familias del Ridge, modelo producto).
- `tests/` — tests con pytest.
- `salidas/` — salida en pantalla del pipeline y de los modelos finales (versionadas).
