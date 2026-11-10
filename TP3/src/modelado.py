"""Piezas comunes a los dos modelos del TP3: carga de datos por pliegue, separación de pliegues
para elegir y para reportar, y evaluación por segmento contra los baselines.

Regla de elección (DECISIONES.md, Etapa A — Modelos): los hiperparámetros se eligen en la primera
mitad de los pliegues y la métrica del informe se reporta en la segunda mitad.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from validacion import CFG as CFG_VAL, dias_disponibles, dias_historia_incompleta, pliegues_walk_forward, verificar

RAIZ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((RAIZ / "config.yaml").read_text())
FEAT_DIR = RAIZ / "data" / "features"
SEMILLA = CFG["semilla"]
HORIZONTES_REPORTE = (1, 3, 6, 12, 24, 48)


def pliegues_eleccion_reporte():
    """(pliegues para elegir hiperparámetros, pliegues para reportar)."""
    pl = pliegues_walk_forward(dias_disponibles())
    verificar(pl, CFG_VAL["gap_dias"])
    mitad = len(pl) // 2
    return list(enumerate(pl[:mitad], 1)), list(enumerate(pl[mitad:], mitad + 1))


def _archivo(dia):
    return FEAT_DIR / f"{dia:%Y_%m_%d}.parquet"


def cargar_entrenamiento(dias, columnas, max_filas=None, semilla=SEMILLA):
    """Filas con target (reales e interpoladas) de los días dados, con una submuestra aleatoria fija
    si superan max_filas. La fracción es la misma para todos los días."""
    max_filas = max_filas or CFG["modelos"]["max_filas_entrenamiento"]
    total = sum(pq.ParquetFile(_archivo(d)).metadata.num_rows for d in dias)
    frac = min(1.0, max_filas / (0.8 * total))         # ~80 % de las filas tiene target
    rng = np.random.default_rng(semilla)
    partes = []
    for d in dias:
        x = pd.read_parquet(_archivo(d), columns=columnas)
        x = x[x["error_c"].notna()]
        if frac < 1:
            x = x[rng.random(len(x)) < frac]
        partes.append(x)
    return pd.concat(partes, ignore_index=True)


def cargar_prueba(dias, columnas):
    """Todas las filas con observación REAL de los días de prueba (las métricas no usan interpoladas)."""
    partes = []
    for d in dias:
        x = pd.read_parquet(_archivo(d), columns=columnas)
        partes.append(x[x["error_c"].notna() & ~x["obs_interpolada"]])
    return pd.concat(partes, ignore_index=True)


def hora_solar(df):
    """Hora solar entera 0-23 a partir de sen/cos (misma definición que los baselines)."""
    ang = np.arctan2(df["hora_solar_sin"].to_numpy(), df["hora_solar_cos"].to_numpy())
    return (np.round(ang / (2 * np.pi) * 24) % 24).astype("int8")


def sesgo_b2(dias_entrenamiento):
    """Baseline B2: error medio por estación × hora solar con observaciones reales del entrenamiento."""
    partes = []
    for d in dias_entrenamiento:
        x = pd.read_parquet(_archivo(d), columns=["station_id", "error_c", "obs_interpolada",
                                                  "hora_solar_sin", "hora_solar_cos"])
        x = x[x["error_c"].notna() & ~x["obs_interpolada"]]
        partes.append(pd.DataFrame({"station_id": x["station_id"].astype(str).to_numpy(), "hora": hora_solar(x),
                                    "error_c": x["error_c"].to_numpy()}))
    a = pd.concat(partes, ignore_index=True)
    return a.groupby(["station_id", "hora"])["error_c"].mean().rename("B2")


def metricas(r):
    return {"n": len(r), "MAE": float(np.abs(r).mean()), "RMSE": float(np.sqrt((r ** 2).mean())), "sesgo": float(r.mean())}


def evaluar(T, predicciones, pliegue):
    """Métricas por segmento y horizonte. T: filas de prueba con error_c, horizonte, dia y B2.
    predicciones: {nombre: corrección predicha}. Todos se miden sobre las mismas filas (B2 definido)."""
    incompletos = set(dias_historia_incompleta())
    inc = T["dia"].isin(incompletos).to_numpy()
    ok = T["B2"].notna().to_numpy()
    filas = []
    for segmento, mseg in (("todas", np.ones(len(T), bool)), ("completa", ~inc), ("incompleta", inc)):
        m = mseg & ok
        if not m.any():
            continue
        e, h = T["error_c"].to_numpy()[m], T["horizonte"].to_numpy()[m]
        for nombre, c in {"B0": np.zeros(len(T)), "B2": T["B2"].to_numpy(), **predicciones}.items():
            r = e - np.asarray(c)[m]
            filas.append({"pliegue": pliegue, "segmento": segmento, "horizonte": "todos", "modelo": nombre, **metricas(r)})
            for hh in HORIZONTES_REPORTE:
                filas.append({"pliegue": pliegue, "segmento": segmento, "horizonte": hh, "modelo": nombre,
                              **metricas(r[h == hh])})
    return filas


def preparar_prueba(dias_prueba, dias_entrenamiento, columnas):
    """Filas reales de prueba con día (texto) y la corrección B2 aprendida en el entrenamiento."""
    T = cargar_prueba(dias_prueba, columnas)
    T["dia"] = T["target_time_utc"].dt.strftime("%Y_%m_%d")
    T["hora"] = hora_solar(T)
    T["sid"] = T["station_id"].astype(str)
    b2 = sesgo_b2(dias_entrenamiento).reset_index().rename(columns={"station_id": "sid"})
    return T.merge(b2, on=["sid", "hora"], how="left")
