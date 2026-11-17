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

Recibe SIEMPRE las 8 familias (17/11/2026): la pregunta del producto es si una caja negra con toda la información
es sustancialmente mejor que el Ridge regularizado de caja de cristal. Los hiperparámetros se eligen una sola vez
(quedan congelados en lgbm_hiperparametros.json; --reelegir los vuelve a elegir) y cada pliegue de reporte queda
en la caché (data/cache/lgbm/): si sus datos no cambiaron, no se vuelve a entrenar.

Uso:  .venv/bin/python src/modelos/lgbm.py [--reelegir]
"""
import itertools
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):                    # corrido como script: src/ tiene que estar en el path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd

import cache
from formato import dec, n as n_
from modelado import CFG, RAIZ, SEMILLA, cargar_entrenamiento, dias_historia_incompleta, evaluar, pliegues_eleccion_reporte, preparar_prueba
from modelos.ridge import CONDICIONES, FAMILIAS, TODAS, fechas_eleccion

P = CFG["modelos"]["lgbm"]
VERSION = 2
X_COLS = [c for cols in FAMILIAS.values() for c in cols]
COLUMNAS = ["station_id", "target_time_utc", "error_c", "obs_interpolada"] + X_COLS
CODIGO = cache.huella_archivos([Path(__file__), Path(__file__).resolve().parents[1] / "modelado.py"])   # editar invalida la caché
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


def elegir(eleccion):
    """Grilla chica (hojas × objetivo) y cantidad de árboles, por MAE con historia completa."""
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
            print(f"  pliegue {k} · {nombre(params)}: {time.time() - t0:.0f} s", flush=True)
        del A, T, XT
    G = pd.DataFrame(grilla)
    crit = G[G.segmento == "completa"].groupby(["config", "arboles"])["MAE"].mean().sort_values()
    conf, n_arb = crit.index[0]
    return G, crit, next(p for p in GRILLA if nombre(p) == conf), int(n_arb)


def pliegue_reporte(k, tr, te, params, n_arb):
    clave = cache.clave(tipo="lgbm", version=VERSION, params=params, arboles=n_arb, fijos={c: P[c] for c in (
        "learning_rate", "min_child_samples", "feature_fraction", "bagging_fraction", "lambda_l2")},
        max_filas=CFG["modelos"]["max_filas_entrenamiento"], semilla=SEMILLA, columnas=X_COLS,
        codigo=CODIGO, entrena=cache.fechas(tr), prueba=cache.fechas(te), huellas=cache.huella_dias(list(tr) + list(te)),
        historia_incompleta=sorted(set(dias_historia_incompleta()) & set(cache.fechas(te))))   # solo los de ESTE pliegue
    res = cache.leer("lgbm", clave)
    if res is not None:
        return res, True
    A = cargar_entrenamiento(tr, COLUMNAS)
    T = preparar_prueba(te, tr, COLUMNAS)
    m = entrenar(A, params, n_arb)
    res = {"filas": evaluar(T, {"lgbm": m.predict(matriz(T))}, k), "modelo": m.model_to_string(),
           "importancia": pd.Series(m.feature_importance("gain"), index=m.feature_name(), name=f"pliegue_{k}"),
           "filas_entrenamiento": len(A)}
    cache.guardar("lgbm", clave, res)
    return res, False


# --- interfaz de producto (la usa src/producto.py) ------------------------------------------------------------
COLUMNA = "lgbm"
REPORTE = RAIZ / "data" / "resultados" / "lgbm_por_pliegue.csv"


def hiperparametros_producto():
    """Los hiperparámetros congelados, si se eligieron con los pliegues de elección actuales."""
    eleccion, _ = pliegues_eleccion_reporte()
    ruta = RAIZ / "data" / "resultados" / "lgbm_hiperparametros.json"
    hp = json.loads(ruta.read_text()) if ruta.exists() else {}
    if hp.get("familias") != TODAS or hp.get("pliegues_eleccion_fechas") != fechas_eleccion(eleccion):
        raise RuntimeError("faltan los hiperparámetros de LightGBM o son viejos: correr src/modelos/lgbm.py")
    return {"num_leaves": hp["num_leaves"], "objective": hp["objective"], "arboles": hp["arboles"],
            "max_filas": CFG["modelos"]["max_filas_entrenamiento"]}


def entrenar_producto(dias, hp):
    A = cargar_entrenamiento(dias, COLUMNAS, hp["max_filas"])
    m = entrenar(A, {"num_leaves": hp["num_leaves"], "objective": hp["objective"]}, hp["arboles"])
    return m.model_to_string(), len(A)                 # texto: se guarda y se carga sin depender de la versión de pickle


def predecir_producto(modelo, df):
    return lgb.Booster(model_str=modelo).predict(matriz(df))


def main():
    salida = RAIZ / "data" / "resultados"
    salida.mkdir(exist_ok=True)
    ruta_hp = salida / "lgbm_hiperparametros.json"
    eleccion, reporte = pliegues_eleccion_reporte()
    print(f"LightGBM con las 8 familias ({len(X_COLS)} variables). Pliegues que eligen: {[k for k, _ in eleccion]} · "
          f"pliegues que reportan: {[k for k, _ in reporte]}", flush=True)

    hp = json.loads(ruta_hp.read_text()) if ruta_hp.exists() else {}
    if "--reelegir" in sys.argv or hp.get("familias") != TODAS or hp.get("pliegues_eleccion_fechas") != fechas_eleccion(eleccion):
        print(f"\nEligiendo hiperparámetros: {len(GRILLA)} combinaciones × cortes de {CORTES[0]} a {CORTES[-1]} árboles "
              "(criterio: MAE con historia completa)...", flush=True)
        G, crit, params, n_arb = elegir(eleccion)
        G.to_csv(salida / "lgbm_eleccion.csv", index=False)
        print("  las 5 mejores:   combinación                      árboles   MAE medio (°C)")
        for (conf, n), v in crit.head(5).items():
            print(f"                  {conf:<32} {n:>7}   {dec(v, 4)}")
        ruta_hp.write_text(json.dumps({**P, **params, "arboles": n_arb, "familias": TODAS,
                                       "pliegues_eleccion": [k for k, _ in eleccion],
                                       "pliegues_eleccion_fechas": fechas_eleccion(eleccion)}))
    else:
        params, n_arb = {"num_leaves": hp["num_leaves"], "objective": hp["objective"]}, hp["arboles"]
        print(f"\nHiperparámetros ya elegidos (congelados en {ruta_hp.relative_to(RAIZ)}): {nombre(params)}, "
              f"{n_arb} árboles. Para volver a elegirlos: --reelegir")

    filas, imp = [], []
    for k, (tr, te) in reporte:
        t0 = time.time()
        res, de_cache = pliegue_reporte(k, tr, te, params, n_arb)
        filas += res["filas"]
        imp.append(res["importancia"])
        (salida / f"lgbm_pliegue_{k}.txt").write_text(res["modelo"])
        print(f"pliegue {k}: " + ("leído de la caché" if de_cache else
              f"entrenado con {n_(res['filas_entrenamiento'])} filas en {time.time() - t0:.0f} s"), flush=True)
    R = pd.DataFrame(filas)
    R.to_csv(salida / "lgbm_por_pliegue.csv", index=False)
    I = pd.concat(imp, axis=1)
    I = I / I.sum()                                         # fracción de la ganancia total por pliegue
    I.to_csv(salida / "lgbm_importancia.csv")

    lin = salida / "interpretable_por_pliegue.csv"
    if lin.exists():
        L = pd.read_csv(lin)
        R = pd.concat([R.astype({"horizonte": str}), L[L.modelo == "lineal"].astype({"horizonte": str})], ignore_index=True)
    modelos = [c for c in ("B0", "B2", "lineal", "lgbm") if c in set(R.modelo)]
    print("\nMAE en °C de la temperatura corregida (más bajo es mejor), promedio de los pliegues de reporte.")
    print("B0 = NWS sin corregir; B2 = error medio de la estación a esa hora; lineal = Ridge; lgbm = este modelo.")
    for seg in ("completa", "incompleta", "todas"):
        x = R[R.segmento == seg].astype({"horizonte": str})
        if x.empty:
            continue
        print(f"\nDías con historia {seg}" if seg != "todas" else "\nTodos los días de prueba",
              f"· pliegues {sorted(int(v) for v in x.pliegue.unique())}")
        t = x.groupby(["horizonte", "modelo"])["MAE"].mean().unstack()
        print(f"  {'horizonte':<10}" + "".join(f"{c:>9}" for c in modelos))
        for h in ["todos", "1", "3", "6", "12", "24", "48"]:
            print(f"  {h + ('' if h == 'todos' else ' h'):<10}" + "".join(f"{dec(t.loc[h, c], 3):>9}" for c in modelos))
    fam = {c: f for f, cols in FAMILIAS.items() for c in cols}
    print("\nImportancia por familia (fracción de la ganancia de los árboles, media de los pliegues):")
    for f_, v in I.mean(axis=1).groupby(fam).sum().sort_values(ascending=False).items():
        print(f"  {f_:<14} {dec(100 * v)} %")


if __name__ == "__main__":
    main()
