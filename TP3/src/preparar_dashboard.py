"""Datos del dashboard (Etapa C): pronóstico del NWS, pronóstico corregido por los dos modelos y su banda de
confianza, para los días de prueba de los pliegues de reporte.

Confianza del pronóstico corregido = banda empírica del error que queda después de corregir:
  - Se calibra con los pliegues de ELECCIÓN (1–2): en cada uno se entrena con su entrenamiento y se mide el
    residuo (observado − corregido) en su prueba. Por horizonte (6 tramos) × condición del tiempo se toman
    los percentiles 10 y 90 (si un grupo tiene menos de MIN_GRUPO filas, se usa solo el tramo de horizonte).
  - Se muestra sobre los pliegues de REPORTE (3–4), que no se usaron para calibrar, y se informa qué
    fracción de las observaciones cayó dentro de la banda (lo esperado: ~80 %).

Salidas en data/dashboard/: filas.parquet (una fila por estación × hora objetivo × horizonte con observación
real), bandas.csv, cobertura.csv y estaciones.parquet.

Uso:  .venv/bin/python src/preparar_dashboard.py   (después de modelo_interpretable.py y modelo_lgbm.py)
"""
import json

import numpy as np
import pandas as pd

import modelo_interpretable as MI
import modelo_lgbm as ML
from modelado import RAIZ, cargar_entrenamiento, dias_historia_incompleta, hora_solar, pliegues_eleccion_reporte, cargar_prueba

RES = RAIZ / "data" / "resultados"
SALIDA = RAIZ / "data" / "dashboard"
TRAMOS = [(1, 3), (4, 6), (7, 12), (13, 24), (25, 36), (37, 48)]
MIN_GRUPO = 2000
MODELOS = ("lineal", "lgbm")


def tramo(h):
    out = np.empty(len(h), dtype=object)
    for a, b in TRAMOS:
        out[(h >= a) & (h <= b)] = f"{a}–{b} h"
    return out


def predecir(pliegues, hp_lin, hp_gbm):
    columnas = list(dict.fromkeys(MI.COLUMNAS + ML.COLUMNAS + ["temp_pron", "lat", "lon"]))
    params = {"num_leaves": hp_gbm["num_leaves"], "objective": hp_gbm["objective"]}
    partes = []
    for k, (tr, te) in pliegues:
        A = cargar_entrenamiento(tr, columnas)
        T = cargar_prueba(te, columnas).reset_index(drop=True)
        dis, m = MI.ajustar(A, hp_lin["sesgo"])
        gbm = ML.entrenar(A, params, hp_gbm["arboles"])
        f = T["temp_pron"].to_numpy(np.float32)
        partes.append(pd.DataFrame({
            "pliegue": np.int8(k), "station_id": T["station_id"].astype(str).to_numpy(),
            "target_time_utc": T["target_time_utc"].to_numpy(), "horizonte": T["horizonte"].to_numpy(np.int8),
            "hora_solar": hora_solar(T), "condicion": T["condicion"].astype(str).to_numpy(),
            "nws": f, "obs": (f + T["error_c"].to_numpy(np.float32)).astype(np.float32),
            "lineal": (f + m.predict(dis.transform(T))).astype(np.float32),
            "lgbm": (f + gbm.predict(ML.matriz(T))).astype(np.float32)}))
        print(f"pliegue {k}: {len(T):,} filas de prueba", flush=True)
        del A, T
    D = pd.concat(partes, ignore_index=True)
    D["tramo"] = tramo(D["horizonte"].to_numpy())
    return D


def calibrar(C):
    """Percentiles 10 y 90 del residuo por modelo × tramo × condición (con respaldo por tramo)."""
    filas = []
    for mod in MODELOS:
        r = C["obs"] - C[mod]
        for (t, c), x in r.groupby([C["tramo"], C["condicion"]]):
            filas.append({"modelo": mod, "tramo": t, "condicion": c, "n": len(x),
                          "q10": x.quantile(0.1), "q90": x.quantile(0.9)})
        for t, x in r.groupby(C["tramo"]):
            filas.append({"modelo": mod, "tramo": t, "condicion": "(todas)", "n": len(x),
                          "q10": x.quantile(0.1), "q90": x.quantile(0.9)})
    B = pd.DataFrame(filas)
    B["ancho"] = B["q90"] - B["q10"]
    return B


def aplicar(D, B):
    """Agrega q10/q90 por fila y modelo; si el grupo tiene pocas filas en la calibración, usa el tramo."""
    for mod in MODELOS:
        b = B[B.modelo == mod]
        fino = b[(b.condicion != "(todas)") & (b.n >= MIN_GRUPO)].set_index(["tramo", "condicion"])[["q10", "q90"]]
        grueso = b[b.condicion == "(todas)"].set_index("tramo")[["q10", "q90"]]
        x = D[["tramo", "condicion"]].join(fino, on=["tramo", "condicion"])
        y = D[["tramo"]].join(grueso, on="tramo")
        for q in ("q10", "q90"):
            D[f"{mod}_{q}"] = x[q].fillna(y[q]).to_numpy(np.float32)
    return D


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    hp_lin = json.loads((RES / "interpretable_hiperparametros.json").read_text())
    hp_gbm = json.loads((RES / "lgbm_hiperparametros.json").read_text())
    eleccion, reporte = pliegues_eleccion_reporte()

    C = predecir(eleccion, hp_lin, hp_gbm)
    B = calibrar(C)
    B.to_csv(SALIDA / "bandas.csv", index=False)
    del C

    D = aplicar(predecir(reporte, hp_lin, hp_gbm), B)
    D["historia"] = np.where(D["target_time_utc"].dt.strftime("%Y_%m_%d").isin(set(dias_historia_incompleta())),
                             "incompleta", "completa")
    cob = []
    for mod in MODELOS:
        dentro = (D["obs"] >= D[mod] + D[f"{mod}_q10"]) & (D["obs"] <= D[mod] + D[f"{mod}_q90"])
        for (p, t), x in dentro.groupby([D["pliegue"], D["tramo"]]):
            cob.append({"modelo": mod, "pliegue": p, "tramo": t, "cobertura": x.mean(), "n": len(x)})
        cob.append({"modelo": mod, "pliegue": 0, "tramo": "todos", "cobertura": dentro.mean(), "n": len(dentro)})
    pd.DataFrame(cob).to_csv(SALIDA / "cobertura.csv", index=False)

    est = pd.concat([pd.read_parquet(p, columns=["station_id", "state", "latitude", "longitude"])
                     .drop_duplicates("station_id") for p in sorted((RAIZ / "data" / "pares").glob("*.parquet"))])
    est = est.assign(station_id=est["station_id"].astype(str)).drop_duplicates("station_id")
    est = est.rename(columns={"state": "estado", "latitude": "lat", "longitude": "lon"})
    est.to_parquet(SALIDA / "estaciones.parquet", index=False)
    for c in ("station_id", "condicion", "tramo", "historia"):
        D[c] = D[c].astype("category")
    D.to_parquet(SALIDA / "filas.parquet", index=False)

    print("\ncobertura de la banda P10–P90 en los pliegues de reporte (esperado ~0,80):")
    print(pd.DataFrame(cob).query("pliegue == 0")[["modelo", "cobertura", "n"]].round(3).to_string(index=False))
    print(f"filas del dashboard: {len(D):,} | estaciones: {D['station_id'].nunique()}")


if __name__ == "__main__":
    main()
