"""Paso 0 del TP3: de los 35 CSV diarios a un parquet por día con el target listo.

Qué hace, en orden:
1. Lee cada CSV diario y guarda solo las columnas que usaremos (parquet por día).
2. Arma la serie de temperatura OBSERVADA por estación y hora (una fila por
   estación-hora; en los CSV la misma observación se repite por cada horizonte).
3. Completa huecos cortos de esa serie por interpolación lineal en el tiempo,
   solo si el hueco dura <= max_horas_hueco. Marca cada valor interpolado.
4. Une la observación al pronóstico y calcula  error = observado - pronosticado.

Las entradas son de solo lectura; todo lo nuevo va a data/.
Uso:  .venv/bin/python src/preparar_datos.py            # incremental
      .venv/bin/python src/preparar_datos.py --rehacer  # todo de cero
"""
from pathlib import Path
import glob
import json
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from formato import n, pct, fecha

RAIZ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((RAIZ / "config.yaml").read_text())
ENTRADA = (RAIZ / CFG["entrada"]).resolve()
SALIDA = RAIZ / CFG["salida"]
MAX_HUECO = CFG["interpolacion"]["max_horas_hueco"]

# Solo pronóstico + lo que se conoce al emitir (nada observado salvo la temperatura-target)
COLUMNAS = [
    "station_id", "state", "latitude", "longitude",
    "forecast_generated_at_utc", "target_time_utc", "actual_lead_hours", "horizon_target_hours",
    "temperature_forecast_c", "relative_humidity_forecast_pct", "dewpoint_forecast_c",
    "wind_speed_forecast_kmh", "wind_direction_forecast", "precip_probability_forecast_pct",
    "short_forecast", "match_type", "temperature_observed_c",
]


def dia_de_archivo(ruta):
    """'YYYY_MM_DD' a partir del nombre. Acepta forecast_observation_2026_10_14.csv y
    forecast_observation_v2_16102026.csv (DDMMYYYY), los dos formatos que hubo en la entrega."""
    nombre = Path(ruta).stem
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})$", nombre)
    if m:
        return "_".join(m.groups())
    m = re.search(r"(\d{2})(\d{2})(\d{4})$", nombre)
    if m:
        return f"{m.group(3)}_{m.group(2)}_{m.group(1)}"
    raise ValueError(f"No se reconoce la fecha en {nombre}")


def leer_dia(csv):
    d = pd.read_csv(csv, usecols=COLUMNAS)
    for c in ("forecast_generated_at_utc", "target_time_utc"):
        d[c] = pd.to_datetime(d[c], utc=True)
    for c in ("latitude", "longitude", "temperature_forecast_c", "relative_humidity_forecast_pct",
              "dewpoint_forecast_c", "wind_speed_forecast_kmh", "precip_probability_forecast_pct",
              "actual_lead_hours", "temperature_observed_c"):
        d[c] = d[c].astype("float32")
    d["horizon_target_hours"] = d["horizon_target_hours"].astype("int8")
    for c in ("station_id", "state", "wind_direction_forecast", "short_forecast", "match_type"):
        d[c] = d[c].astype("category")
    return d


def interpolar_huecos_cortos(serie, max_hueco):
    """Interpola linealmente SOLO los huecos de <= max_hueco horas seguidas.

    serie: Series con índice horario regular y NaN donde falta el dato.
    Devuelve (serie_completada, bool_interpolada).
    pandas.interpolate(limit=3) rellenaría los primeros 3 puntos de un hueco largo,
    que es justo lo que NO queremos; por eso medimos el largo de cada hueco.
    """
    es_nan = serie.isna()
    # id de racha: cambia cada vez que se pasa de dato a hueco o viceversa
    racha = (es_nan != es_nan.shift()).cumsum()
    largo = es_nan.groupby(racha).transform("sum")           # largo del hueco al que pertenece
    interior = serie.interpolate(method="linear", limit_area="inside")  # rellena todo el interior
    corto = es_nan & (largo <= max_hueco)
    completa = serie.where(~corto, interior)
    return completa, corto & completa.notna()


def main(rehacer=False):
    """Incremental: solo lee los CSV que todavía no se procesaron.

    - dias_crudos/<dia>.parquet : pronósticos del día (sin la observación)
    - obs_dia/<dia>.parquet     : observaciones válidas y únicas del día
    - pares/<dia>.parquet       : pronóstico + observación (real o interpolada) + target
    La interpolación se rehace siempre sobre la grilla completa (es barata); un día de
    pares/ se vuelve a escribir solo si cambió la observación de ese día (por ejemplo,
    el último día anterior cuando llega un día nuevo y se puede cerrar un hueco).
    """
    for sub in ("dias_crudos", "obs_dia", "pares"):
        (SALIDA / sub).mkdir(parents=True, exist_ok=True)
    dias_dir, obs_dir, pares_dir = SALIDA / "dias_crudos", SALIDA / "obs_dia", SALIDA / "pares"
    archivos = sorted(glob.glob(str(ENTRADA / "forecast_observation_2026_*.csv"))
                      + glob.glob(str(ENTRADA / "forecast_observation_v2_*.csv")), key=dia_de_archivo)
    dias = [dia_de_archivo(f) for f in archivos]
    repetidos = {x for x in dias if dias.count(x) > 1}
    assert not repetidos, f"días con más de un archivo: {sorted(repetidos)}"
    fechas = pd.to_datetime(dias, format="%Y_%m_%d")
    faltan = pd.date_range(fechas.min(), fechas.max()).difference(fechas)
    print("Lee los CSV diarios (pronóstico del NWS + temperatura observada), une cada pronóstico con lo que")
    print("realmente se midió y calcula el error de temperatura: error = observado − pronosticado, en °C.\n")
    print(f"Archivos de entrada: {len(archivos)} días, del {fecha(fechas.min())} al {fecha(fechas.max())}.")
    if len(faltan):
        print(f"AVISO: no hay archivo para {len(faltan)} días del período: "
              f"{', '.join(x.strftime('%d/%m') for x in faltan)}. Esos días quedan fuera del análisis.")

    # --- Paso 1 y 2: parquet por día + observaciones únicas (solo días nuevos) ---------
    nuevos = 0
    for f in archivos:
        dia = dia_de_archivo(f)
        if not rehacer and (dias_dir / f"{dia}.parquet").exists() and (obs_dir / f"{dia}.parquet").exists():
            continue
        d = leer_dia(f)
        # la observación sólo es confiable si cayó DENTRO del período del pronóstico
        valida = (d["match_type"] == "interval") & d["temperature_observed_c"].notna()
        o = d.loc[valida, ["station_id", "target_time_utc", "temperature_observed_c"]].copy()
        o["station_id"] = o["station_id"].astype(str)
        o.drop_duplicates(["station_id", "target_time_utc"]).to_parquet(obs_dir / f"{dia}.parquet", index=False)
        d.drop(columns=["temperature_observed_c"]).to_parquet(dias_dir / f"{dia}.parquet", index=False)
        nuevos += 1
        print(f"  día nuevo {dia[8:]}/{dia[5:7]}/{dia[:4]}: {n(len(d))} filas")
    print(f"Días leídos en esta corrida: {nuevos} (los demás ya estaban procesados de corridas anteriores).")

    obs = pd.concat([pd.read_parquet(p) for p in sorted(obs_dir.glob("*.parquet"))], ignore_index=True)
    # control: una observación por estación-hora (si hubiera valores distintos, avisamos)
    dup = obs.duplicated(["station_id", "target_time_utc"]).sum()
    obs = obs.drop_duplicates(["station_id", "target_time_utc"])

    # --- Paso 3: grilla horaria completa por estación + interpolación ------------------
    horas = pd.date_range(obs["target_time_utc"].min(), obs["target_time_utc"].max(), freq="h")
    estaciones = sorted(obs["station_id"].unique())
    grilla = pd.MultiIndex.from_product([estaciones, horas], names=["station_id", "target_time_utc"])
    serie = obs.set_index(["station_id", "target_time_utc"])["temperature_observed_c"].reindex(grilla)

    partes = []
    for est, s in serie.groupby(level="station_id"):
        s = s.droplevel("station_id")
        completa, interpolada = interpolar_huecos_cortos(s, MAX_HUECO)
        partes.append(pd.DataFrame({"station_id": est, "target_time_utc": s.index,
                                    "temp_obs_c": completa.values.astype("float32"),
                                    "obs_interpolada": interpolada.values}))
    obs_h = pd.concat(partes, ignore_index=True)
    obs_h.to_parquet(SALIDA / "obs_horaria.parquet", index=False)
    real, interp, sin = serie.notna().sum(), obs_h.obs_interpolada.sum(), obs_h.temp_obs_c.isna().sum()
    tot = len(obs_h)
    print(f"\nTemperatura observada: una lectura por estación y hora ({len(estaciones)} estaciones × "
          f"{n(len(horas))} horas = {n(tot)} lecturas posibles)")
    print(f"  con lectura real del NWS:      {n(real):>9}  ({pct(real / tot)})")
    print(f"  completadas por interpolación: {n(interp):>9}  ({pct(interp / tot)})  "
          f"huecos de hasta {MAX_HUECO} h entre dos lecturas reales")
    print(f"  sin dato:                      {n(sin):>9}  ({pct(sin / tot)})  "
          f"huecos más largos: esas horas quedan sin error")
    print(f"  lecturas repetidas entre archivos de días distintos: {n(dup)}"
          + ("" if dup == 0 else " (AVISO: se conservó la primera)"))

    # --- Paso 4: unir al pronóstico y calcular el target (solo días que cambiaron) -----
    obs_h["dia"] = obs_h["target_time_utc"].dt.strftime("%Y_%m_%d")
    huellas = {dia: int(pd.util.hash_pandas_object(g[["station_id", "temp_obs_c", "obs_interpolada"]], index=False).sum())
               for dia, g in obs_h.groupby("dia")}
    manifiesto = SALIDA / "manifiesto_pares.json"
    previas = json.loads(manifiesto.read_text()) if manifiesto.exists() and not rehacer else {}
    obs_h["station_id"] = obs_h["station_id"].astype("category")
    hechos = 0
    for p in sorted(dias_dir.glob("*.parquet")):
        dia = p.stem
        if previas.get(dia) == huellas.get(dia) and (pares_dir / p.name).exists():
            continue
        d = pd.read_parquet(p)
        d["station_id"] = d["station_id"].astype(str).astype(obs_h["station_id"].dtype)
        d = d.merge(obs_h.drop(columns="dia"), on=["station_id", "target_time_utc"], how="left")
        d["obs_interpolada"] = d["obs_interpolada"].fillna(False)
        d["error_c"] = (d["temp_obs_c"] - d["temperature_forecast_c"]).astype("float32")  # obs - pron
        d.to_parquet(pares_dir / p.name, index=False)
        hechos += 1
    manifiesto.write_text(json.dumps(huellas))
    total = sum(pq.ParquetFile(p).metadata.num_rows for p in pares_dir.glob("*.parquet"))
    print(f"\nTabla pronóstico-observación (una fila por estación × hora objetivo × horizonte de 1 a 48 h): "
          f"{n(total)} filas.")
    print(f"Días recalculados en esta corrida: {hechos} (solo los nuevos o aquellos cuya observación cambió).")


if __name__ == "__main__":
    main(rehacer="--rehacer" in sys.argv)
