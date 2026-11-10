"""Modelo que prioriza minimizar el error: LightGBM con todas las familias de features.

Recibe todo lo que el interpretable deja afuera (humedad, punto de rocío, dirección del viento, latitud,
longitud, los cuatro sesgos exponenciales). Los árboles encuentran solos las interacciones (por ejemplo,
noche × despejado × calma) y manejan los faltantes de forma nativa: no se imputa nada.
No se usa station_id: el sesgo propio de cada estación ya entra por los sesgos exponenciales.

Regla (DECISIONES.md, Etapa A — Modelos):
  1. En la primera mitad de los pliegues se elige una combinación de una grilla chica (hojas × objetivo)
     y la cantidad de árboles (cortes cada PASO_ARBOLES), por MAE con historia completa.
  2. Con eso fijo se reentrena en cada pliegue de la segunda mitad y se reporta contra B0, B2 y el
     modelo interpretable (si ya se corrió: se lee de data/resultados/interpretable_por_pliegue.csv).

Uso:  .venv/bin/python src/modelo_lgbm.py
"""
import itertools
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from modelado import CFG, RAIZ, SEMILLA, cargar_entrenamiento, evaluar, pliegues_eleccion_reporte, preparar_prueba
from modelo_interpretable import CONDICIONES

P = CFG["modelos"]["lgbm"]
FAMILIAS = json.loads((RAIZ / "data" / "features" / "familias.json").read_text())
X_COLS = [c for cols in FAMILIAS.values() for c in cols]
COLUMNAS = ["station_id", "target_time_utc", "error_c", "obs_interpolada"] + X_COLS
GRILLA = [dict(num_leaves=h, objective=o) for h, o in itertools.product(P["num_leaves"], P["objetivos"])]
CORTES = list(range(P["paso_arboles"], P["max_arboles"] + 1, P["paso_arboles"]))


def matriz(df):
    X = df[X_COLS].copy()
    X["condicion"] = pd.Categorical(X["condicion"].astype(str), categories=CONDICIONES)
    for c in ("noche", "viento_calmo"):
        X[c] = X[c].astype("int8")
    return X


def entrenar(A, params, n_arboles):
    base = dict(learning_rate=P["learning_rate"], min_child_samples=P["min_child_samples"],
                feature_fraction=P["feature_fraction"], bagging_fraction=P["bagging_fraction"], bagging_freq=1,
                lambda_l2=P["lambda_l2"], seed=SEMILLA, deterministic=True, force_row_wise=True, verbose=-1)
    ds = lgb.Dataset(matriz(A), A["error_c"].to_numpy(), categorical_feature=["condicion"], free_raw_data=True)
    return lgb.train({**base, **params}, ds, num_boost_round=n_arboles)


def nombre(params):
    return f"hojas{params['num_leaves']}_{params['objective']}"


def main():
    salida = RAIZ / "data" / "resultados"
    salida.mkdir(exist_ok=True)
    eleccion, reporte = pliegues_eleccion_reporte()
    print("eligen:", [k for k, _ in eleccion], "| reportan:", [k for k, _ in reporte], "| grilla:", len(GRILLA),
          "| cortes:", CORTES, flush=True)

    # --- 1. elección en la primera mitad ----------------------------------------------------
    grilla = []
    for k, (tr, te) in eleccion:
        A = cargar_entrenamiento(tr, COLUMNAS)
        T = preparar_prueba(te, tr, COLUMNAS)
        XT = matriz(T)
        for params in GRILLA:
            t0 = time.time()
            m = entrenar(A, params, P["max_arboles"])
            pred = {f"{nombre(params)}|{n}": m.predict(XT, num_iteration=n) for n in CORTES}
            for f in evaluar(T, pred, k):
                if f["modelo"] not in ("B0", "B2") and f["horizonte"] == "todos":
                    conf, n = f["modelo"].split("|")
                    grilla.append({**f, "config": conf, "arboles": int(n)})
            print(f"pliegue {k} · {nombre(params)}: {time.time() - t0:.0f} s", flush=True)
        del A, T, XT
    G = pd.DataFrame(grilla)
    G.to_csv(salida / "lgbm_eleccion.csv", index=False)
    crit = G[G.segmento == "completa"].groupby(["config", "arboles"])["MAE"].mean().sort_values()
    print("\nMAE medio (historia completa) en los pliegues de elección, 10 mejores:")
    print(crit.head(10).round(4).to_string())
    conf, n_arb = crit.index[0]
    params = next(p for p in GRILLA if nombre(p) == conf)
    print("elegido:", conf, "con", n_arb, "árboles", flush=True)

    # --- 2. reporte en la segunda mitad ------------------------------------------------------
    filas, imp = [], []
    for k, (tr, te) in reporte:
        t0 = time.time()
        A = cargar_entrenamiento(tr, COLUMNAS)
        T = preparar_prueba(te, tr, COLUMNAS)
        m = entrenar(A, params, n_arb)
        filas += evaluar(T, {"lgbm": m.predict(matriz(T))}, k)
        imp.append(pd.Series(m.feature_importance("gain"), index=m.feature_name(), name=f"pliegue_{k}"))
        m.save_model(str(salida / f"lgbm_pliegue_{k}.txt"))
        print(f"pliegue {k}: {len(A):,} entrenamiento | {len(T):,} prueba | {time.time() - t0:.0f} s", flush=True)
        del A, T
    R = pd.DataFrame(filas)
    R.to_csv(salida / "lgbm_por_pliegue.csv", index=False)
    I = pd.concat(imp, axis=1)
    I = I / I.sum()                                         # fracción de la ganancia total por pliegue
    I.to_csv(salida / "lgbm_importancia.csv")
    (salida / "lgbm_hiperparametros.json").write_text(json.dumps({**P, **params, "arboles": int(n_arb)}))

    lin = salida / "interpretable_por_pliegue.csv"
    if lin.exists():
        L = pd.read_csv(lin)
        L = L[L.modelo == "lineal"].astype({"horizonte": str})
        R = pd.concat([R.astype({"horizonte": str}), L], ignore_index=True)
    pd.set_option("display.width", 200)
    for seg in ("completa", "incompleta", "todas"):
        x = R[R.segmento == seg]
        if x.empty:
            continue
        print(f"\n== MAE · segmento {seg} · pliegues {sorted(int(v) for v in x.pliegue.unique())}")
        t = x.groupby(["horizonte", "modelo"])["MAE"].mean().unstack()
        print(t.reindex(["todos"] + [str(h) for h in (1, 3, 6, 12, 24, 48)])[[c for c in ("B0", "B2", "lineal", "lgbm") if c in t]].round(3))
    print("\nimportancia (fracción de ganancia, media de pliegues):")
    print(I.mean(axis=1).sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
