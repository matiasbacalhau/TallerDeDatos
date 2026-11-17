"""Modelo explicativo alternativo: regresión lineal SIN penalización (OLS) con selección por RFE.

Es un experimento independiente del Ridge del producto (DECISIONES.md, «Experimento: OLS con RFE»). Mismo dataset y
misma matriz de diseño (Disenio de modelo_interpretable.py con las 8 familias: curvas spline, interacciones e
indicadores de faltante), pero la selección la hace Recursive Feature Elimination y no el encogimiento del Ridge.
No usa nada que haya decidido el Ridge: ni su α, ni la ablación por familias, ni sus 5 familias.

  1. Unidad que se elimina: la VARIABLE de features.py con todas sus columnas de diseño (bases spline, indicador
     falta_, interacciones). Ver variable_de().
  2. RFE en los pliegues de elección: se ajusta OLS con las variables vivas, se mide el peso de cada una
     (desvío de su aporte Σβx en el ENTRENAMIENTO, °C), se promedia entre pliegues y sale la de menor peso.
     Se repite hasta que queda una. Resultado: un solo orden de eliminación y conjuntos anidados.
  3. Curva: MAE (historia completa) de cada conjunto en cada pliegue de elección.
  4. Regla (mismos umbrales del grupo, experimentos.umbral_*_c): el conjunto más chico que, contra el mejor punto de
     la curva, no empeora más de 0,005 °C de promedio ni 0,010 °C en ningún pliegue.
  5. Reporte: OLS con ese conjunto en los pliegues de reporte, contra B0 y B2. La comparación con Ridge y LightGBM
     se hace recién en notebooks/rfe_ols.ipynb.

OLS sale de la misma matriz XᵀX que usa ridge_gram.Gram (con α = 0). La matriz es singular: las bases spline y los
8 indicadores de condición suman 1 y, centrados, quedan colineales. Por eso se resuelve con la seudoinversa
(solución de norma mínima): la predicción y el aporte de cada variable no dependen de cuál solución se tome, porque
la colinealidad es interna a cada variable.

Uso:  .venv/bin/python src/modelo_rfe.py
"""
import json
import re

import numpy as np
import pandas as pd

import cache
import experimentos as E
import modelo_interpretable as MI
from modelado import RAIZ, evaluar, pliegues_eleccion_reporte

SALIDA = RAIZ / "salidas" / "rfe"
TOL_AUTOVALOR = 1e-10                      # autovalores < TOL · máximo se tratan como 0 (colinealidad exacta)
PROPIAS = {"enfriamiento_nocturno": "base", "hora_solar": "base"}     # variables que arma Disenio, no features.py


# --- variables ---------------------------------------------------------------------------------------------------
def variable_de_columna(nombre):
    """Variable de features.py (o derivada en Disenio) a la que pertenece una columna de diseño."""
    if nombre == "enfriamiento_nocturno":
        return nombre
    if nombre.startswith("cond_"):
        return "condicion"
    if re.fullmatch(r"hora_s\d+", nombre):
        return "hora_solar"
    if nombre.startswith("err_ayer_x_"):
        return "err_ayer"
    m = re.fullmatch(r"(.+)_x_h_s\d+", nombre)
    if m:
        return m.group(1)
    if nombre.endswith("_x_noche"):
        return nombre[: -len("_x_noche")]
    if nombre.startswith("falta_"):
        return nombre[len("falta_"):]
    m = re.fullmatch(r"(.+)_s\d+", nombre)
    if m and m.group(1) in MI.CURVAS:
        return m.group(1)
    return nombre


def variable_de(dis):
    """Array con la variable de cada columna de la matriz de diseño (mismo orden que dis.nombres)."""
    v = np.array([variable_de_columna(c) for c in dis.nombres])
    desconocidas = set(v) - set(MI.FAMILIA_DE) - set(PROPIAS)
    assert not desconocidas, f"columnas sin variable conocida: {desconocidas}"
    return v


def familia_de_variable(v):
    return PROPIAS.get(v) or MI.FAMILIA_DE[v]


# --- OLS desde XᵀX -----------------------------------------------------------------------------------------------
def ols(gram, mascara):
    """(coeficientes con 0 fuera de la máscara, intercepto). Solución de norma mínima (seudoinversa)."""
    m = np.asarray(mascara, bool)
    coef = np.zeros(len(m))
    lam, V = np.linalg.eigh(gram.xtx[np.ix_(m, m)])
    ok = lam > TOL_AUTOVALOR * lam.max()
    coef[m] = V[:, ok] @ ((V[:, ok].T @ gram.xty[m]) / lam[ok])
    return coef, float(gram.media_y - gram.media_x @ coef)


def pesos(gram, n, coef, var, vivas):
    """Desvío (°C) del aporte Σβx de cada variable viva en el entrenamiento: √(β_gᵀ XᵀX_gg β_g / n)."""
    out = {}
    for v in vivas:
        g = var == v
        b = coef[g]
        out[v] = float(np.sqrt(max(b @ gram.xtx[np.ix_(g, g)] @ b, 0.0) / n))
    return out


def rfe(grams):
    """RFE con un solo orden para todos los pliegues. grams: lista de (gram, n, var) con el mismo orden de columnas.
    Devuelve (pasos: una fila por paso con las vivas y la que sale, coeficientes por paso y pliegue)."""
    var = grams[0][2]
    vivas = list(dict.fromkeys(var))
    pasos, coefs = [], []
    while vivas:
        mascara = np.isin(var, vivas)
        sol = [ols(g, mascara) for g, _, _ in grams]
        P = pd.DataFrame([pesos(g, n, c, v, vivas) for (g, n, v), (c, _) in zip(grams, sol)])
        prom = P.mean()
        sale = prom.idxmin()
        pasos.append({"n_variables": len(vivas), "vivas": list(vivas), "sale": sale, "peso_sale_C": prom[sale],
                      "pesos": prom.to_dict()})
        coefs.append(sol)
        vivas = [v for v in vivas if v != sale]
    return pasos, coefs


def elegir(curva, umbral_prom=E.CFG_E["umbral_promedio_c"], umbral_pliegue=E.CFG_E["umbral_pliegue_c"]):
    """curva: DataFrame índice n_variables, una columna por pliegue (MAE). Devuelve (n elegido, n del mejor punto,
    tabla con los deltas contra el mejor punto y si cumple la regla)."""
    mejor = curva.mean(axis=1).idxmin()
    D = curva - curva.loc[mejor]
    tab = pd.DataFrame({"delta_promedio": D.mean(axis=1), "delta_max": D.max(axis=1)})
    tab["cumple"] = [E.aplicar_regla(p, m, umbral_prom, umbral_pliegue) for p, m in zip(tab.delta_promedio, tab.delta_max)]
    return int(tab.index[tab.cumple].min()), int(mejor), tab


# --- datos por pliegue -------------------------------------------------------------------------------------------
def preparar(k, tr, te):
    """(Disenio de 8 familias, Gram, filas de entrenamiento, filas de prueba) del pliegue; Gram de la caché."""
    dis, g, T = E.preparar_gram(k, tr, te)
    n = cache.leer("gram", MI.clave_pliegue("gram", MI.TODAS, None, tr, te, MI.P["max_filas"]))["filas"]
    return dis, g, n, T


def curvas(dis, coef, variables, puntos=60):
    """Aporte (°C, centrado) de cada curva viva sobre una grilla de su variable, en la escala original."""
    filas = []
    idx = {c: i for i, c in enumerate(dis.nombres)}
    mu, sd = dis.esc.mean_, dis.esc.scale_
    fuentes = {c: (sp, lambda x, c=c: pd.DataFrame({c: x})) for c, sp in dis.sp.items()}
    if dis.base:
        fuentes["hora_solar"] = (dis.sp_hora, lambda x: x.reshape(-1, 1))
    for v, (sp, armar) in fuentes.items():
        if v not in variables:
            continue
        nombres = [f"hora_s{i}" if v == "hora_solar" else f"{v}_s{i}" for i in range(sp.n_features_out_)]
        j = [idx[c] for c in nombres]
        lo, hi = (0.0, 24.0) if v == "hora_solar" else (sp.bsplines_[0].t[sp.degree], sp.bsplines_[0].t[-sp.degree - 1])
        x = np.linspace(lo, hi, puntos)
        a = ((sp.transform(armar(x)) - mu[j]) / sd[j]) @ coef[j]
        filas += [{"variable": v, "x": xi, "aporte_C": ai} for xi, ai in zip(x, a - a.mean())]
    return filas


# --- main --------------------------------------------------------------------------------------------------------
def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    eleccion, _ = E.pliegues()
    print("OLS + RFE. Pliegues de elección:", [k for k, _ in eleccion], flush=True)

    datos = []
    for k, (tr, te) in eleccion:
        dis, g, n, T = preparar(k, tr, te)
        datos.append((k, dis, g, n, T))
        print(f"  pliegue {k}: {n:,} filas de entrenamiento, {len(dis.nombres)} columnas de diseño", flush=True)
    var = variable_de(datos[0][1])
    for _, dis, *_ in datos[1:]:
        assert list(dis.nombres) == list(datos[0][1].nombres), "los pliegues no tienen las mismas columnas"

    pasos, coefs = rfe([(g, n, var) for _, _, g, n, _ in datos])
    curva = {}
    for j, (k, dis, g, n, T) in enumerate(datos):
        C = np.column_stack([coefs[p][j][0] for p in range(len(pasos))])
        b = np.array([coefs[p][j][1] for p in range(len(pasos))])
        curva[k] = E._mae_por_columna(dis, C, b, T)
    curva = pd.DataFrame(curva, index=[p["n_variables"] for p in pasos]).rename_axis("n_variables")
    curva.columns = [f"MAE_pliegue_{k}" for k in curva.columns]
    n_elegido, n_mejor, tab = elegir(curva)
    elegidas = next(p["vivas"] for p in pasos if p["n_variables"] == n_elegido)

    orden = pd.DataFrame([{"paso": i + 1, "n_variables": p["n_variables"], "sale": p["sale"],
                           "familia": familia_de_variable(p["sale"]), "peso_sale_C": p["peso_sale_C"]}
                          for i, p in enumerate(pasos)])
    orden.to_csv(SALIDA / "orden_eliminacion.csv", index=False)
    pd.DataFrame([p["pesos"] for p in pasos], index=orden.n_variables).to_csv(SALIDA / "pesos_por_paso.csv")
    curva.join(tab).to_csv(SALIDA / "curva.csv")
    (SALIDA / "conjunto_elegido.json").write_text(json.dumps({
        "n_variables": n_elegido, "n_variables_mejor_punto": n_mejor, "variables": elegidas,
        "familias": {v: familia_de_variable(v) for v in elegidas},
        "pliegues_eleccion_fechas": MI.fechas_eleccion(eleccion),
        "umbral_promedio_c": E.CFG_E["umbral_promedio_c"], "umbral_pliegue_c": E.CFG_E["umbral_pliegue_c"]},
        indent=1, ensure_ascii=False))
    print(f"\nCurva (MAE medio, elección): todas ({len(pasos)}) = {curva.iloc[0].mean():.4f} · "
          f"mejor ({n_mejor}) = {curva.loc[n_mejor].mean():.4f} · elegido ({n_elegido}) = {curva.loc[n_elegido].mean():.4f}")
    print("Variables elegidas:", ", ".join(elegidas), flush=True)
    del datos

    _, reporte = pliegues_eleccion_reporte()
    filas, por_dia, coef_rep, curv = [], [], {}, []
    for k, (tr, te) in reporte:
        dis, g, n, T = preparar(k, tr, te)
        v = variable_de(dis)
        coef, b0 = ols(g, np.isin(v, elegidas))
        T = T[T["B2"].notna()].reset_index(drop=True)                     # mismas filas que evaluar()
        pred = MI.predecir(dis, coef, b0, T)
        filas += evaluar(T, {"ols_rfe": pred}, k)
        ae = pd.DataFrame({"dia": T["dia"].to_numpy(), "ols_rfe": np.abs(T["error_c"].to_numpy() - pred)})
        por_dia.append(ae.groupby("dia").agg(n=("ols_rfe", "size"), ols_rfe=("ols_rfe", "mean")).assign(pliegue=k))
        coef_rep[f"pliegue_{k}"] = pd.Series(coef, index=dis.nombres)
        curv += [{**f, "pliegue": k} for f in curvas(dis, coef, elegidas)]
        P = pesos(g, n, coef, v, elegidas)
        coef_rep[f"peso_pliegue_{k}"] = pd.Series({c: P[x] for c, x in zip(dis.nombres, v) if x in P})
        print(f"  reporte, pliegue {k}: {len(T):,} filas de prueba", flush=True)
        del T
    R = pd.DataFrame(filas)
    R.to_csv(SALIDA / "por_pliegue.csv", index=False)
    pd.concat(por_dia).reset_index()[["pliegue", "dia", "n", "ols_rfe"]].to_csv(SALIDA / "por_dia.csv", index=False)
    C = pd.DataFrame(coef_rep)
    C.insert(0, "variable", [variable_de_columna(c) for c in C.index])
    C.rename_axis("columna").to_csv(SALIDA / "coeficientes.csv")
    pd.DataFrame(curv).to_csv(SALIDA / "curvas.csv", index=False)

    x = R[(R.segmento == "todas")].astype({"horizonte": str}).groupby(["horizonte", "modelo"])["MAE"].mean().unstack()
    print("\nMAE (°C), pliegues de reporte, todos los días de prueba:")
    print(x.loc[["todos", "1", "6", "24", "48"], ["B0", "B2", "ols_rfe"]].round(3).to_string())
    print(f"\nSalidas en {SALIDA.relative_to(RAIZ)}/")


if __name__ == "__main__":
    main()
