"""Punto de rocío OBSERVADO por estación y hora, para la familia 8 de features.py (err_rocio_*).

Lee de los CSV diarios del TP1 (solo lectura) las columnas mínimas y aplica el mismo criterio de
validez que preparar_datos.py para la temperatura: la observación solo vale si cayó DENTRO del
período del pronóstico (match_type == "interval"). Sin interpolar: solo valores reales.

Salida: data/obs_rocio.parquet  (station_id, target_time_utc, dewpoint_observed_c)
Uso:    .venv/bin/python src/rocio_observado.py
"""
import glob

import pandas as pd
import pyarrow.csv as pv

from preparar_datos import CFG, ENTRADA, SALIDA, dia_de_archivo

COLS = ["station_id", "target_time_utc", "match_type", "dewpoint_observed_c"]


def main():
    archivos = sorted(glob.glob(str(ENTRADA / "forecast_observation_2026_*.csv"))
                      + glob.glob(str(ENTRADA / "forecast_observation_v2_*.csv")), key=dia_de_archivo)
    partes = []
    for f in archivos:
        t = pv.read_csv(f, convert_options=pv.ConvertOptions(include_columns=COLS)).to_pandas()
        t = t[(t["match_type"] == "interval") & t["dewpoint_observed_c"].notna()]
        t["station_id"] = t["station_id"].astype(str)
        t["target_time_utc"] = pd.to_datetime(t["target_time_utc"], utc=True)
        partes.append(t[COLS[:2] + COLS[3:]].drop_duplicates(["station_id", "target_time_utc"]))
        print(" ", dia_de_archivo(f), len(partes[-1]), "observaciones de rocío")
    o = pd.concat(partes, ignore_index=True).drop_duplicates(["station_id", "target_time_utc"])
    o["dewpoint_observed_c"] = o["dewpoint_observed_c"].astype("float32")
    o.to_parquet(SALIDA / "obs_rocio.parquet", index=False)
    print("listo:", len(o), "estación-hora con rocío observado")


if __name__ == "__main__":
    main()
