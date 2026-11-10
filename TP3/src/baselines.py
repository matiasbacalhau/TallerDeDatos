"""Baselines del TP3: el piso de performance que cualquier modelo debe superar.

Cada baseline propone una corrección c; el pronóstico corregido es  pronóstico + c.
Como  error = observado - pronosticado,  el error que queda es  error - c.

  B0  NWS crudo            c = 0
  B1  sesgo global         c = error medio del entrenamiento
  B2  sesgo estación×hora  c = error medio de la estación a esa hora solar (entrenamiento)
  B3  persistencia         c = último error conocido de la estación, mismo horizonte

Todo lo que se aprende (B1, B2) usa SOLO observaciones reales del entrenamiento.
Se evalúa en dos universos: filas con observación real y filas reales + interpoladas.
Para que la comparación sea justa, todos se miden sobre las mismas filas: aquellas
donde B2 y B3 están definidos.

Segmentos (columna `segmento`): «todas»; «completa», días de prueba cuyos días previos están
todos (la comparación limpia); «incompleta», días que siguen a un hueco o a días excluidos
(robustez ante caídas). Ver src/calidad_dias.py.

Uso:  .venv/bin/python src/baselines.py
"""
from pathlib import Path
import glob

import numpy as np
import pandas as pd

from validacion import CFG as CFG_VAL, dias_disponibles, dias_historia_incompleta, pliegues_walk_forward, verificar

RAIZ = Path(__file__).resolve().parents[1]
COLUMNAS = ["station_id", "longitude", "target_time_utc", "forecast_generated_at_utc",
            "horizon_target_hours", "error_c", "obs_interpolada"]
# Una observación de la hora H se registra hasta H+59 min. Para estar seguros de que
# existía al emitir, el último error usable es el de la hora objetivo - horizonte - 2.
RETRASO_PERSISTENCIA_H = 2


def cargar_pares():
    partes = [pd.read_parquet(f, columns=COLUMNAS) for f in sorted(glob.glob(str(RAIZ / "data/pares/*.parquet")))]
    d = pd.concat(partes, ignore_index=True)
    d["station_id"] = d["station_id"].astype(str)
    d["h"] = d["horizon_target_hours"].astype("int16")
    return d


def agregar_variables(d):
    """Hora solar, día y último error causal (para B3). No toca el target."""
    d["dia"] = d["target_time_utc"].dt.tz_convert(None).dt.normalize()
    # hora solar = hora UTC + longitud/15 (sin saltos por cambio de hora civil)
    d["hora_solar"] = ((d["target_time_utc"].dt.hour + np.round(d["longitude"] / 15)).astype(int) % 24).astype("int8")
    # error de la misma estación y mismo horizonte, en la hora objetivo - h - 2 (solo observaciones reales)
    real = d.loc[d["error_c"].notna() & ~d["obs_interpolada"], ["station_id", "target_time_utc", "h", "error_c"]]
    real = real.rename(columns={"target_time_utc": "t_fuente", "error_c": "e_ultimo"})
    d["t_fuente"] = d["target_time_utc"] - pd.to_timedelta(d["h"] + RETRASO_PERSISTENCIA_H, unit="h")
    d = d.merge(real, on=["station_id", "t_fuente", "h"], how="left")
    return d


def prueba_causalidad(d):
    """La hora de la que sale e_ultimo tiene que estar completa antes de la emisión."""
    f = d.loc[d["e_ultimo"].notna()]
    # una observación de la hora H puede llegar hasta H + 59 min
    disponible = f["t_fuente"] + pd.Timedelta(minutes=59)
    viola = (disponible > f["forecast_generated_at_utc"]).sum()
    assert viola == 0, f"{viola} filas usan una observación posterior a la emisión"
    return len(f)


def metricas(r):
    return pd.Series({"n": len(r), "MAE": r.abs().mean(), "RMSE": np.sqrt((r ** 2).mean()), "sesgo": r.mean()})


def evaluar(d):
    pliegues = pliegues_walk_forward(dias_disponibles())
    verificar(pliegues, CFG_VAL["gap_dias"])
    incompletos = pd.to_datetime(sorted(dias_historia_incompleta()), format="%Y_%m_%d")
    filas = []
    for k, (tr, te) in enumerate(pliegues, 1):
        A = d[d["dia"].isin(tr) & d["error_c"].notna() & ~d["obs_interpolada"]]
        T = d[d["dia"].isin(te) & d["error_c"].notna()].copy()
        T["B0"] = 0.0
        T["B1"] = A["error_c"].mean()
        T = T.join(A.groupby(["station_id", "hora_solar"])["error_c"].mean().rename("B2"), on=["station_id", "hora_solar"])
        T["B3"] = T["e_ultimo"]
        comun = T["B2"].notna() & T["B3"].notna()          # mismas filas para todos
        inc = T["dia"].isin(incompletos)
        for segmento, mseg in (("todas", True), ("completa", ~inc), ("incompleta", inc)):
            for universo, mascara in (("reales", ~T["obs_interpolada"]), ("reales+interpoladas", True)):
                S = T[comun & mascara & mseg]
                if S.empty:
                    continue
                for b in ("B0", "B1", "B2", "B3"):
                    r = S["error_c"] - S[b]
                    filas.append({"pliegue": k, "segmento": segmento, "universo": universo, "horizonte": "todos",
                                  "baseline": b, **metricas(r)})
                    for hh in (1, 3, 6, 12, 24, 48):
                        rr = (S["error_c"] - S[b])[S["h"] == hh]
                        filas.append({"pliegue": k, "segmento": segmento, "universo": universo, "horizonte": hh,
                                      "baseline": b, **metricas(rr)})
            ms = T[mseg] if segmento != "todas" else T
            cs = comun[mseg] if segmento != "todas" else comun
            filas.append({"pliegue": k, "segmento": segmento, "universo": "cobertura_comun", "horizonte": "todos",
                          "baseline": "-", "n": cs.sum(), "MAE": cs.mean() if len(ms) else np.nan,
                          "RMSE": np.nan, "sesgo": np.nan})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    d = agregar_variables(cargar_pares())
    n = prueba_causalidad(d)
    print(f"prueba de causalidad OK ({n:,} filas con último error)")
    R = evaluar(d)
    salida = RAIZ / "data" / "resultados"
    salida.mkdir(exist_ok=True)
    R.to_csv(salida / "baselines_por_pliegue.csv", index=False)
    r = R[R["universo"] != "cobertura_comun"]
    resumen = (r.groupby(["segmento", "universo", "horizonte", "baseline"])[["MAE", "RMSE", "sesgo"]]
               .agg(["mean"]).droplevel(1, axis=1).round(3).reset_index())
    n_pl = r.groupby("segmento")["pliegue"].nunique().rename("n_pliegues").reset_index()
    resumen = resumen.merge(n_pl, on="segmento")
    resumen.to_csv(salida / "baselines_resumen.csv", index=False)
    pd.set_option("display.width", 200)
    for seg in ("completa", "incompleta", "todas"):
        x = resumen[(resumen.segmento == seg) & (resumen.universo == "reales")]
        if x.empty:
            continue
        print(f"\n== MAE medio por pliegue · segmento: {seg} ({x.n_pliegues.iloc[0]} pliegues) · universo: reales")
        print(x.pivot(index="horizonte", columns="baseline", values="MAE").reindex(["todos", 1, 3, 6, 12, 24, 48]))
        print(x[x.horizonte == "todos"].set_index("baseline")[["RMSE", "sesgo"]].T)
    print("\nfilas comunes (n por pliegue / fracción):")
    print(R[R.universo == "cobertura_comun"][["pliegue", "segmento", "n", "MAE"]]
          .rename(columns={"MAE": "fracción"}).to_string(index=False))
