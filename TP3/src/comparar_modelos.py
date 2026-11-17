"""¿Las diferencias de error entre modelos son mayores que el ruido? (pliegues de reporte)

Reentrena los dos modelos con los hiperparámetros ya elegidos (data/resultados/*_hiperparametros.json) en
cada pliegue de reporte y compara el MAE POR DÍA de prueba.

Unidad de la prueba: el DÍA. Dentro de un día los errores de las estaciones y de los horizontes están muy
correlacionados (la misma situación del tiempo afecta a toda la región), así que tratar cada fila como
independiente daría p-valores falsamente chicos. Con D_d = MAE_A(d) − MAE_B(d):
  - t pareado sobre los D_d (prueba de Diebold–Mariano con pérdida absoluta, sin corrección por
    autocorrelación entre días: con 10 días no se puede estimar);
  - intervalo bootstrap del 95 % remuestreando días;
  - cuántos días gana cada modelo.

Usa los hiperparámetros congelados y los modelos LightGBM ya entrenados (data/resultados/lgbm_pliegue_k.txt).
Las predicciones por día de cada pliegue quedan en la caché (data/cache/comparacion/).

Uso:  .venv/bin/python src/comparar_modelos.py
"""
import json

import lightgbm as lgb

import numpy as np
import pandas as pd
from scipy import stats

import modelo_interpretable as MI
import modelo_lgbm as ML
import cache
from modelado import RAIZ, SEMILLA, cargar_entrenamiento, dias_historia_incompleta, pliegues_eleccion_reporte, preparar_prueba

PARES = [("lgbm", "lineal"), ("lineal", "B2"), ("lgbm", "B2"), ("B2", "B0")]
N_BOOT = 10000


def predicciones_por_dia(reporte):
    res_dir = RAIZ / "data" / "resultados"
    hp_lin = json.loads((res_dir / "interpretable_hiperparametros.json").read_text())
    hp_gbm = json.loads((res_dir / "lgbm_hiperparametros.json").read_text())
    fams, alpha = hp_lin["familias"], hp_lin["ridge_alpha"]
    columnas = list(dict.fromkeys(MI.columnas_de(fams) + ML.COLUMNAS))
    filas = []
    for k, (tr, te) in reporte:
        modelo_gbm = (res_dir / f"lgbm_pliegue_{k}.txt").read_text()
        clave = MI.clave_pliegue("comparacion", fams, alpha, tr, te, MI.P["max_filas"])
        clave = cache.clave(base=clave, lgbm=modelo_gbm)
        agg = cache.leer("comparacion", clave)
        if agg is None:
            A = cargar_entrenamiento(tr, MI.columnas_de(fams), MI.P["max_filas"])   # la misma submuestra del Ridge
            dis, m = MI.ajustar(A, fams, alpha)
            del A
            T = preparar_prueba(te, tr, columnas)
            T = T[T["B2"].notna()].reset_index(drop=True)           # mismas filas que evaluar()
            pred = {"B0": np.zeros(len(T)), "B2": T["B2"].to_numpy(), "lineal": MI.predecir(dis, m.coef_, m.intercept_, T),
                    "lgbm": lgb.Booster(model_str=modelo_gbm).predict(ML.matriz(T))}
            e = T["error_c"].to_numpy()
            ae = pd.DataFrame({n: np.abs(e - p) for n, p in pred.items()})
            ae["dia"], ae["pliegue"] = T["dia"].to_numpy(), k
            agg = ae.groupby(["pliegue", "dia"]).agg(n=("B0", "size"), **{n: (n, "mean") for n in pred})
            cache.guardar("comparacion", clave, agg)
            print(f"pliegue {k}: {len(T):,} filas de prueba", flush=True)
        else:
            print(f"pliegue {k}: leído de la caché", flush=True)
        filas.append(agg)
    D = pd.concat(filas).reset_index()
    D["historia"] = np.where(D["dia"].isin(set(dias_historia_incompleta())), "incompleta", "completa")
    return D


def prueba(d, rng):
    n = len(d)
    t, p = stats.ttest_1samp(d, 0.0)
    boot = rng.choice(d, size=(N_BOOT, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"dias": n, "dif_media_C": d.mean(), "IC95_inf": lo, "IC95_sup": hi, "t": t, "p_valor": p,
            "dias_gana_A": int((d < 0).sum())}


def main():
    _, reporte = pliegues_eleccion_reporte()
    D = predicciones_por_dia(reporte)
    salida = RAIZ / "data" / "resultados"
    D.to_csv(salida / "comparacion_por_dia.csv", index=False)
    rng = np.random.default_rng(SEMILLA)
    res = []
    for seg, x in [("todas", D)] + list(D.groupby("historia")):
        for a, b in PARES:
            res.append({"segmento": seg, "A": a, "B": b, **prueba((x[a] - x[b]).to_numpy(), rng)})
    R = pd.DataFrame(res)
    R.to_csv(salida / "comparacion_significancia.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nMAE por día (°C):")
    print(D.round(3).to_string(index=False))
    print("\nA − B en MAE diario (negativo = A mejor). IC bootstrap por día; t pareado sobre días:")
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
