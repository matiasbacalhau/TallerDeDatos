"""Dashboard del TP3: corrección de pronósticos de temperatura a 24 horas.

Ubicar este archivo junto a la carpeta resultados_tp3 creada por el notebook.
Ejecutar: python -m streamlit run dashboard_tp3.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="TP3 | Corrección meteorológica",
    page_icon="🌦️",
    layout="wide",
)

CARPETA_BASE = Path(__file__).resolve().parent
ARCHIVO = CARPETA_BASE / "resultados_tp3" / "predicciones_test.csv"
ERRORES = {
    "Pronóstico original": "error_original_c",
    "Ridge": "error_ridge_c",
    "Gradient Boosting": "error_hgb_c",
}
COLUMNAS_OBLIGATORIAS = {
    "station_id", "state", "latitude", "longitude", "target_time_utc",
    "temperature_forecast_c", "temperature_observed_c",
    "temperature_ridge_c", "temperature_hgb_c", *ERRORES.values(),
}


@st.cache_data(show_spinner="Leyendo predicciones exportadas...")
def cargar_datos(ruta: str, modificado: float) -> pd.DataFrame:
    # modificado permite invalidar la caché si se reemplaza el CSV.
    datos = pd.read_csv(ruta, low_memory=False)
    faltantes = COLUMNAS_OBLIGATORIAS.difference(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en predicciones_test.csv: {sorted(faltantes)}")
    datos["target_time_utc"] = pd.to_datetime(
        datos["target_time_utc"], utc=True, errors="coerce"
    )
    datos = datos.dropna(subset=["target_time_utc", *ERRORES.values()]).copy()
    datos["fecha"] = datos["target_time_utc"].dt.date
    datos["hora_utc"] = datos["target_time_utc"].dt.hour
    datos["state"] = datos["state"].fillna("Sin estado").astype(str)
    datos["station_id"] = datos["station_id"].astype(str)
    return datos


def resumen_errores(datos: pd.DataFrame) -> pd.DataFrame:
    filas = []
    mae_original = datos[ERRORES["Pronóstico original"]].abs().mean()
    for nombre, columna in ERRORES.items():
        error = datos[columna].to_numpy(dtype=float)
        mae = np.mean(np.abs(error))
        filas.append({
            "Modelo": nombre,
            "MAE (°C)": mae,
            "RMSE (°C)": np.sqrt(np.mean(error ** 2)),
            "Sesgo (°C)": np.mean(error),
            "Mejora MAE (%)": 100 * (1 - mae / mae_original) if mae_original else np.nan,
        })
    return pd.DataFrame(filas)


st.title("🌦️ Corrección de pronósticos meteorológicos")
st.caption(
    "TP3 · Temperatura · Horizonte de 24 horas · Conjunto de prueba "
    "(23 de octubre al 1 de noviembre de 2026)"
)

if not ARCHIVO.is_file():
    st.error("No encuentro el archivo de predicciones exportadas.")
    st.code(str(ARCHIVO), language="text")
    st.info(
        "Guardá dashboard_tp3.py en la carpeta que contiene resultados_tp3/ "
        "y verificá que allí esté predicciones_test.csv."
    )
    st.stop()

try:
    df = cargar_datos(str(ARCHIVO), ARCHIVO.stat().st_mtime)
except (ValueError, OSError, pd.errors.ParserError) as error:
    st.error(f"No fue posible cargar el CSV: {error}")
    st.stop()

if df.empty:
    st.warning("El archivo no contiene registros válidos.")
    st.stop()

# Filtros interactivos. Todas las métricas se recalculan con la misma selección.
st.sidebar.header("Filtros")
fecha_min, fecha_max = df["fecha"].min(), df["fecha"].max()
periodo = st.sidebar.date_input(
    "Fechas del objetivo (UTC)",
    value=(fecha_min, fecha_max),
    min_value=fecha_min,
    max_value=fecha_max,
)
if isinstance(periodo, tuple) and len(periodo) == 2:
    inicio, fin = periodo
else:
    st.info("Seleccioná una fecha de inicio y otra de fin en la barra lateral.")
    st.stop()

estados = sorted(df["state"].unique().tolist())
estados_seleccionados = st.sidebar.multiselect(
    "Estados (sin selección = todos)", estados
)
mascara = (df["fecha"] >= inicio) & (df["fecha"] <= fin)
if estados_seleccionados:
    mascara &= df["state"].isin(estados_seleccionados)

estaciones_disponibles = sorted(df.loc[mascara, "station_id"].unique().tolist())
estaciones_seleccionadas = st.sidebar.multiselect(
    "Estaciones (sin selección = todas)", estaciones_disponibles
)
if estaciones_seleccionadas:
    mascara &= df["station_id"].isin(estaciones_seleccionadas)

filtrado = df.loc[mascara].copy()
if filtrado.empty:
    st.warning("No hay registros para los filtros seleccionados.")
    st.stop()

metricas = resumen_errores(filtrado)
mae_original = metricas.iloc[0]["MAE (°C)"]
mae_ridge = metricas.iloc[1]["MAE (°C)"]
mae_hgb = metricas.iloc[2]["MAE (°C)"]
mejora_hgb = metricas.iloc[2]["Mejora MAE (%)"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE original", f"{mae_original:.3f} °C")
c2.metric("MAE Ridge", f"{mae_ridge:.3f} °C")
c3.metric("MAE Gradient Boosting", f"{mae_hgb:.3f} °C")
c4.metric("Mejora de Gradient Boosting", f"{mejora_hgb:+.2f} %")
st.caption(
    f"{len(filtrado):,} pronósticos válidos · "
    f"{filtrado['station_id'].nunique()} estaciones · "
    f"{filtrado['fecha'].nunique()} días · Horarios UTC"
)

pestana_resumen, pestana_tiempo, pestana_estaciones, pestana_datos = st.tabs(
    ["Resumen", "Evolución temporal", "Estaciones y mapa", "Datos y descarga"]
)

with pestana_resumen:
    st.subheader("Comparación de modelos")
    st.bar_chart(metricas.set_index("Modelo")[["MAE (°C)"]])
    st.dataframe(metricas.round(4), hide_index=True, use_container_width=True)
    st.info(
        "El error se define como observado − pronosticado. "
        "El MAE y el RMSE más bajos indican mayor precisión; "
        "el sesgo más próximo a cero indica menor error medio firmado."
    )

with pestana_tiempo:
    st.subheader("MAE diario")
    diario = filtrado.groupby("fecha", as_index=False).agg(
        registros=("station_id", "size"),
        estaciones=("station_id", "nunique"),
        mae_original=("error_original_c", lambda x: x.abs().mean()),
        mae_ridge=("error_ridge_c", lambda x: x.abs().mean()),
        mae_hgb=("error_hgb_c", lambda x: x.abs().mean()),
        sesgo_hgb=("error_hgb_c", "mean"),
    )
    diario["fecha"] = pd.to_datetime(diario["fecha"])
    diario["mejora_hgb_pct"] = 100 * (
        1 - diario["mae_hgb"] / diario["mae_original"]
    )
    st.line_chart(
        diario.set_index("fecha")[["mae_original", "mae_ridge", "mae_hgb"]]
        .rename(columns={
            "mae_original": "Original",
            "mae_ridge": "Ridge",
            "mae_hgb": "Gradient Boosting",
        })
    )
    izq, der = st.columns(2)
    with izq:
        st.subheader("Estaciones por día")
        st.bar_chart(diario.set_index("fecha")[["estaciones"]])
    with der:
        st.subheader("Sesgo diario del modelo HGB")
        st.line_chart(diario.set_index("fecha")[["sesgo_hgb"]])
    st.dataframe(diario.round(4), hide_index=True, use_container_width=True)
    st.subheader("Sesgo según hora UTC")
    sesgo_horario = filtrado.groupby("hora_utc")[
        ["error_original_c", "error_hgb_c"]
    ].mean().rename(columns={
        "error_original_c": "Original", "error_hgb_c": "Gradient Boosting"
    })
    st.line_chart(sesgo_horario)
    st.caption("La hora es UTC, no la hora local de cada estación.")

with pestana_estaciones:
    st.subheader("Ubicación de estaciones con observaciones")
    mapa = filtrado[["station_id", "latitude", "longitude"]].drop_duplicates(
        subset=["station_id"]
    ).dropna(subset=["latitude", "longitude"])
    mapa = mapa[
        mapa["latitude"].between(-90, 90)
        & mapa["longitude"].between(-180, 180)
    ]
    if len(mapa):
        st.map(mapa[["latitude", "longitude"]])
    else:
        st.info("No hay coordenadas válidas en la selección.")

    st.subheader("MAE por estado")
    por_estado = filtrado.groupby("state", as_index=False).agg(
        registros=("station_id", "size"),
        estaciones=("station_id", "nunique"),
        mae_original=("error_original_c", lambda x: x.abs().mean()),
        mae_hgb=("error_hgb_c", lambda x: x.abs().mean()),
    )
    por_estado["mejora_hgb_pct"] = 100 * (
        1 - por_estado["mae_hgb"] / por_estado["mae_original"]
    )
    por_estado = por_estado.sort_values("registros", ascending=False)
    st.dataframe(por_estado.round(3), hide_index=True, use_container_width=True)
    st.caption(
        "La comparación entre estados puede estar afectada por diferencias "
        "de cobertura y cantidad de registros."
    )

with pestana_datos:
    st.subheader("Explorar pronósticos individuales")
    columnas_visibles = [
        "station_id", "state", "target_time_utc",
        "temperature_forecast_c", "temperature_observed_c",
        "temperature_ridge_c", "temperature_hgb_c",
        "error_original_c", "error_ridge_c", "error_hgb_c",
    ]
    st.dataframe(
        filtrado[columnas_visibles].head(1000).round(3),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Vista limitada a 1.000 filas; la descarga incluye todas las filas filtradas.")
    st.download_button(
        "Descargar CSV filtrado",
        data=filtrado[columnas_visibles].to_csv(index=False).encode("utf-8"),
        file_name="predicciones_tp3_filtradas.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Resultados descriptivos de prueba. El dashboard no reentrena ni modifica modelos. "
    "Se consideraron solo observaciones emparejadas con match_type='interval' "
    "para el horizonte de 24 h. La cobertura cambia entre el 23 y el 26 de octubre."
)
