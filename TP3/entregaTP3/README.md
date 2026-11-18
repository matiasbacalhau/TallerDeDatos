# Entrega TP3 — Grupo 3 (US East)

Corrección con ML del error de temperatura de los pronósticos horarios del NWS (`error = observado − pronosticado`), 516 estaciones del Este de EE.UU., horizontes de 1 a 48 h, 55 días (10/09 → 09/11/2026).

Esta carpeta es el **índice de la entrega**. El código no está copiado acá: cada punto enlaza al archivo del repo que lo implementa, para que haya una sola versión de cada cosa.

| Punto de la consigna | Entregable | Cómo correrlo (desde `TP3/`) | Qué produce |
|---|---|---|---|
| **1. Modelo que prioriza la interpretabilidad** | [`src/modelos/ridge.py`](../src/modelos/ridge.py) — Ridge con curvas (splines), 5 familias de variables | `.venv/bin/python src/modelos/ridge.py` | [`salidas/ridge/`](../salidas/ridge/): α elegido, grilla, MAE/RMSE/sesgo por pliegue, horizonte y segmento, coeficientes |
| **2. Modelo que prioriza minimizar el error** | [`src/modelos/lgbm.py`](../src/modelos/lgbm.py) — LightGBM, 8 familias (51 variables) | `.venv/bin/python src/modelos/lgbm.py` | [`salidas/lgbm/`](../salidas/lgbm/): grilla elegida, métricas por pliegue, importancias |
| **3. Informe** | [`informe_tp3.pdf`](informe_tp3.pdf) (fuente: [`informe_tp3.qmd`](informe_tp3.qmd); gráficos y tablas de [`notebooks/comparacion_modelos.ipynb`](../notebooks/comparacion_modelos.ipynb)) | `QUARTO_PYTHON=.venv/bin/python quarto render entregaTP3/informe_tp3.qmd` | PDF (las tablas se calculan desde `salidas/`) |
| **4. Dashboard** | [`app/dashboard.py`](../app/dashboard.py) — ver [`dashboard.md`](dashboard.md) | `.venv-app/bin/streamlit run app/dashboard.py` | App en http://localhost:8501 |

## Pipeline de entrenamiento, validación cruzada y métricas (puntos 1 y 2)

Los dos scripts siguen el mismo esquema (`main()` de cada uno):

1. **Datos y variables:** [`src/modelado.py`](../src/modelado.py) carga cada pliegue con las variables de [`src/features.py`](../src/features.py). Todas son conocidas al emitir el pronóstico; lo controla [`src/verificar_features.py`](../src/verificar_features.py).
2. **Validación cruzada walk-forward** ([`src/validacion.py`](../src/validacion.py)): ventana expansiva, cortes por día, 2 días de gap, 5 pliegues de 5 días anclados al 30/09/2026. Nunca k-fold con mezcla.
3. **Elección de hiperparámetros en los pliegues 1–2** (Ridge: α; LightGBM: hojas, pérdida y cantidad de árboles). Quedan congelados en `salidas/<modelo>/hiperparametros.json`.
4. **Métricas en los pliegues 3–5**, que no se usaron para elegir: MAE, RMSE y sesgo por horizonte y por segmento (días con historia completa o incompleta), contra los baselines B0 (NWS crudo) y B2 (sesgo por estación × hora solar) de [`src/baselines.py`](../src/baselines.py).

La comparación entre los dos modelos, con el día como unidad (t pareado y bootstrap), está en [`src/comparar_modelos.py`](../src/comparar_modelos.py). Las figuras de interpretación (curvas del Ridge y SHAP de LightGBM) las genera [`src/interpretar.py`](../src/interpretar.py) en [`figuras/`](../figuras/).

Salida en pantalla de la última corrida de todo esto: [`salidas/modelos_finales.txt`](../salidas/modelos_finales.txt).

## Reproducir desde cero

Hace falta el dataset del TP1 (CSV diarios, ~700 MB, no está en el repo). Pasos completos en el [README de TP3](../README.md#cómo-reproducir). Para ver el dashboard **no** hacen falta los datos: ver [`dashboard.md`](dashboard.md).

Decisiones y su justificación: [`DECISIONES.md`](../DECISIONES.md).
