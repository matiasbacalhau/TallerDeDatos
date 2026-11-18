# Punto 4 — Dashboard

Código: [`app/dashboard.py`](../app/dashboard.py) (Streamlit + Plotly). Los datos que usa están versionados en [`app/datos/`](../app/datos/) (~26 MB), así que **no hace falta el dataset crudo ni correr el pipeline**.

## Correrlo

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

Se abre en http://localhost:8501. La primera carga tarda unos segundos (6,8 millones de filas).

## Qué muestra

| Pestaña | Contenido |
|---|---|
| Hallazgos | La narrativa: qué corrige el modelo y cuánto |
| Resumen | Indicadores: error del NWS, error corregido, mejora y cobertura de la banda |
| Ridge (Gram) / LightGBM | Versiones, variables, búsqueda de hiperparámetros y regularización de cada modelo |
| Ridge vs LightGBM | Comparación con filtros de horizonte, días de prueba, condición y estados |
| **Monitoreo** | Pronóstico del NWS vs. corregido vs. observado por estación, con banda P10–P90; error por horizonte y por hora; mapa de mejora por estación |
| Variables | Figuras de interpretación (curvas del Ridge, SHAP de LightGBM) |
| Datos y validación | Días excluidos, pliegues walk-forward, baselines B0–B3 |

**Filtros** (pedidos por la consigna): corrección (Ridge o LightGBM), horizonte, hora solar local, condición del tiempo y estados.

Los datos son solo los días de prueba de los pliegues de reporte (3–5), que los modelos no vieron al entrenar.

Para regenerar `app/datos/` con días nuevos: pipeline completo, después `src/preparar_dashboard.py` y `src/preparar_resultados_dashboard.py`.
