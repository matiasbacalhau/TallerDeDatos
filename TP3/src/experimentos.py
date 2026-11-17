"""Experimentación por familias: decide qué familias usa el Ridge (caja de cristal) antes de entrenar el modelo final.

Recorrido (notebooks/experimentos.ipynb lo presenta y lo corre):
  1. hipótesis experta por familia (en la notebook);
  2. exploración con los días de exploración: correlación de cada variable con el error del NWS, redundancia entre
     familias y boxplots;
  3. Ridge con todas las familias: α elegido en los pliegues de elección, aporte de cada familia y su estabilidad;
  4. Ridge sin cada familia (ablación) y REGLA DE RECORTE fijada de antemano (DECISIONES.md, 17/11/2026): una
     familia se saca si sin ella el MAE promedio de los pliegues de elección no empeora en más de umbral_promedio_c
     (0,005 °C) y en ningún pliegue de elección empeora en más de umbral_pliegue_c (0,010 °C).

Datos congelados: solo días hasta experimentos.fecha_corte (09/11/2026). Las decisiones usan SOLO los pliegues de
elección; la exploración, solo días hasta el final del último bloque de prueba de elección. Ningún día de prueba de
los pliegues de reporte interviene: si interviniera, el resultado final quedaría optimista.

Todas las variantes del Ridge salen de UNA matriz XᵀX por pliegue (src/ridge_gram.py): sacar una familia es sacar
sus filas y columnas, con el mismo resultado que reentrenar. Las matrices quedan en la caché (data/cache/gram/).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import cache
import modelos.ridge as MI
from modelado import CFG, RAIZ, SEMILLA, _archivo, cargar_entrenamiento, dias_historia_incompleta, evaluar, preparar_prueba
from ridge_gram import Gram
from validacion import CFG as CFG_VAL, dias_disponibles, pliegues_walk_forward, verificar

CFG_E = CFG["experimentos"]
CORTE = pd.Timestamp(CFG_E["fecha_corte"])
SALIDA = RAIZ / "salidas" / "experimentos"
HORIZONTES = (1, 6, 24, 48)


# --- pliegues y días ---------------------------------------------------------------------------------------------
def pliegues(dias=None, hasta=CORTE):
    """(pliegues de elección, pliegues de reporte) con el corte congelado."""
    pl = pliegues_walk_forward(dias_disponibles() if dias is None else dias, hasta=hasta)
    verificar(pl, CFG_VAL["gap_dias"])
    ne = CFG_VAL["n_pliegues_eleccion"]
    return list(enumerate(pl[:ne], 1)), list(enumerate(pl[ne:], ne + 1))


def dias_exploracion(dias=None, hasta=CORTE):
    """Días usables hasta el último día de prueba del último pliegue de elección."""
    dias = dias_disponibles() if dias is None else pd.DatetimeIndex(dias)
    eleccion, _ = pliegues(dias, hasta)
    return dias[dias <= eleccion[-1][1][1].max()]


# --- exploración -------------------------------------------------------------------------------------------------
def muestra_exploracion(n=500_000):
    """Filas con observación real de los días de exploración (submuestra fija), con todas las variables."""
    dias = dias_exploracion()
    cols = ["station_id", "target_time_utc", "error_c", "obs_interpolada"] + [c for cs in MI.FAMILIAS.values() for c in cs]
    total = sum(len(pd.read_parquet(_archivo(d), columns=["error_c"])) for d in dias)
    frac = min(1.0, n / (0.75 * total))
    rng = np.random.default_rng(SEMILLA)
    partes = []
    for d in dias:
        x = pd.read_parquet(_archivo(d), columns=list(dict.fromkeys(cols)))
        x = x[x["error_c"].notna() & ~x["obs_interpolada"]]
        partes.append(x[rng.random(len(x)) < frac])
    return pd.concat(partes, ignore_index=True)


def variables_numericas():
    return [c for cs in MI.FAMILIAS.values() for c in cs if c != "condicion"]


def correlaciones(df):
    """Spearman de cada variable con el error del NWS: en total y por horizonte."""
    filas = []
    r = df[variables_numericas() + ["error_c"]].astype("float64").rank()
    for c in variables_numericas():
        fila = {"variable": c, "familia": MI.FAMILIA_DE[c], "faltantes_pct": 100 * df[c].isna().mean(),
                "total": r[c].corr(r["error_c"])}
        for h in HORIZONTES:
            m = (df["horizonte"] == h).to_numpy()
            rh = df.loc[m, [c, "error_c"]].astype("float64").rank()
            fila[f"h{h}"] = rh[c].corr(rh["error_c"])
        filas.append(fila)
    return pd.DataFrame(filas)


def mapa_correlaciones(df):
    """Spearman entre todas las variables numéricas (para el mapa de calor)."""
    return df[variables_numericas()].astype("float64").corr(method="spearman")


def redundancia_entre_familias(M):
    """Para cada par de familias, la mayor |correlación| entre una variable de una y una de la otra."""
    fams = MI.TODAS
    R = pd.DataFrame(np.nan, index=fams, columns=fams)
    for a in fams:
        for b in fams:
            ca = [c for c in MI.FAMILIAS[a] if c in M.index]
            cb = [c for c in MI.FAMILIAS[b] if c in M.index]
            sub = M.loc[ca, cb].abs()
            if a == b:
                sub = sub.where(~np.eye(len(ca), dtype=bool))
            R.loc[a, b] = sub.max().max()
    return R


# --- Ridge: una XᵀX por pliegue ----------------------------------------------------------------------------------
def preparar_gram(k, tr, te):
    """(Disenio con las 8 familias, Gram, filas de prueba) de un pliegue de elección; Disenio y Gram en la caché."""
    clave = MI.clave_pliegue("gram", MI.TODAS, None, tr, te, MI.P["max_filas"])
    res = cache.leer("gram", clave)
    if res is None:
        A = cargar_entrenamiento(tr, MI.columnas_de(MI.TODAS), MI.P["max_filas"])
        dis = MI.Disenio(MI.TODAS).fit(A)
        res = {"dis": dis, "gram": Gram.desde(dis.transform(A), A["error_c"].to_numpy()), "filas": len(A)}
        del A
        cache.guardar("gram", clave, res)
    T = preparar_prueba(te, tr, MI.columnas_de(MI.TODAS))
    return res["dis"], res["gram"], T


def mascara(dis, familias):
    return np.isin(np.array(dis.familia_de), list(familias))


def variantes_ablacion():
    """«todas» y una variante sin cada familia."""
    return {"todas": MI.TODAS, **{f"sin {f}": [x for x in MI.TODAS if x != f] for f in MI.TODAS}}


def _mae_por_columna(dis, C, b, T):
    """MAE (historia completa, B2 definido) de cada columna de predicción, por bloques de filas."""
    ok = (~T["dia"].isin(set(dias_historia_incompleta())) & T["B2"].notna()).to_numpy()
    e = T["error_c"].to_numpy()
    suma, n = np.zeros(C.shape[1]), 0
    for i in range(0, len(T), MI.BLOQUE):
        m = ok[i:i + MI.BLOQUE]
        X = dis.transform(T.iloc[i:i + MI.BLOQUE][m])
        suma += np.abs(e[i:i + MI.BLOQUE][m, None] - (X @ C + b)).sum(axis=0)
        n += m.sum()
    return suma / n


def tabla_variantes(variantes, datos, alphas=None):
    """MAE (historia completa) de cada variante × α × pliegue de elección. Si el mejor α de alguna variante queda en
    un borde de la grilla, se agrega un orden de magnitud hacia ese lado (como máximo dos veces)."""
    grilla = list(alphas or MI.alphas())
    filas = []
    for _ in range(3):
        hechos = {f["alpha"] for f in filas}
        nuevos = [a for a in grilla if a not in hechos]
        for k, dis, g, T in datos:
            combos = [(v, a) for v in variantes for a in nuevos]
            sol = [g.resolver(mascara(dis, variantes[v]), a) for v, a in combos]
            mae = _mae_por_columna(dis, np.column_stack([c for c, _ in sol]), np.array([b for _, b in sol]), T)
            filas += [{"variante": v, "alpha": a, "pliegue": k, "MAE": m} for (v, a), m in zip(combos, mae)]
        tab = pd.DataFrame(filas)
        mejores = tab.groupby(["variante", "alpha"])["MAE"].mean().groupby("variante").idxmin().map(lambda t: t[1])
        bordes = set(mejores) & {min(grilla), max(grilla)}
        if not bordes:
            break
        grilla += [a / 10 if a == min(grilla) else a * 10 for a in bordes]
    return tab


def mejor_por_variante(tab):
    """Por variante: el α de menor MAE medio y el MAE por pliegue con ese α."""
    medio = tab.groupby(["variante", "alpha"])["MAE"].mean()
    elegido = medio.groupby("variante").idxmin()
    filas = []
    for v, (_, a) in elegido.items():
        x = tab[(tab.variante == v) & (tab.alpha == a)].set_index("pliegue")["MAE"]
        filas.append({"variante": v, "alpha": a, "MAE_medio": x.mean(), **{f"MAE_pliegue_{k}": m for k, m in x.items()}})
    return pd.DataFrame(filas).set_index("variante")


def aplicar_regla(delta_promedio, delta_max, umbral_prom=CFG_E["umbral_promedio_c"], umbral_pliegue=CFG_E["umbral_pliegue_c"]):
    """True = la familia se saca: sin ella el modelo no empeora de forma distinguible (regla del 17/11/2026)."""
    return bool(delta_promedio <= umbral_prom + 1e-12 and delta_max <= umbral_pliegue + 1e-12)


def ablacion(mejores):
    """Una fila por familia: cuánto empeora el MAE al sacarla (con su propio α) y la decisión de la regla."""
    cols = [c for c in mejores.columns if c.startswith("MAE_pliegue_")]
    base = mejores.loc["todas", cols]
    filas = []
    for f in MI.TODAS:
        d = mejores.loc[f"sin {f}", cols] - base
        filas.append({"familia": f, "alpha_sin": mejores.loc[f"sin {f}", "alpha"], "delta_promedio": d.mean(),
                      "delta_max": d.max(), **{f"delta_{c[4:]}": v for c, v in d.items()},
                      "se_saca": aplicar_regla(d.mean(), d.max())})
    return pd.DataFrame(filas).set_index("familia")


def confirmar_recorte(a_sacar, costo, evaluar_recorte, umbral_prom=CFG_E["umbral_promedio_c"]):
    """Saca juntas las familias que la regla saca una por una. Si juntas empeoran más que el umbral (dos familias que
    se reemplazan entre sí), devuelve al modelo la de mayor costo de sacarla y vuelve a probar.
    evaluar_recorte(lista) -> empeoramiento promedio (°C) contra el Ridge con todas."""
    sacar = list(a_sacar)
    while sacar and evaluar_recorte(sacar) > umbral_prom + 1e-12:
        sacar.remove(max(sacar, key=lambda f: costo[f]))
    return sacar


def contribuciones_todas(datos, alpha):
    """Por familia y pliegue de elección: desvío del aporte de la familia a la predicción (°C) con el Ridge con todas,
    y qué fracción de sus coeficientes cambia de signo entre pliegues (estabilidad)."""
    filas, signos = [], {}
    for k, dis, g, T in datos:
        coef, _ = g.resolver(np.ones(len(dis.nombres), bool), alpha)
        fam = np.array(dis.familia_de)
        s1, s2, n = {f: 0.0 for f in dis.familias}, {f: 0.0 for f in dis.familias}, 0
        for i in range(0, len(T), MI.BLOQUE):
            X = dis.transform(T.iloc[i:i + MI.BLOQUE])
            for f in dis.familias:
                a = X[:, fam == f] @ coef[fam == f]
                s1[f] += a.sum()
                s2[f] += (a.astype(np.float64) ** 2).sum()
            n += len(X)
        for f in dis.familias:
            filas.append({"familia": f, "pliegue": k, "aporte_desvio_C": np.sqrt(max(s2[f] / n - (s1[f] / n) ** 2, 0))})
        signos[k] = pd.Series(np.sign(coef), index=dis.nombres)
    S = pd.concat(signos, axis=1)
    cambia = (S.nunique(axis=1) > 1).groupby(pd.Series(dis.familia_de, index=dis.nombres)).mean()
    C = pd.DataFrame(filas)
    C["coef_que_cambian_de_signo_pct"] = C["familia"].map(100 * cambia)
    return C


def evaluar_detalle(variantes, datos):
    """Filas de evaluar() (MAE por segmento y horizonte) en los pliegues de elección.
    variantes: {nombre: (familias, α)}."""
    filas = []
    for k, dis, g, T in datos:
        sol = [g.resolver(mascara(dis, fams), a) for fams, a in variantes.values()]
        P = MI.predecir(dis, np.column_stack([c for c, _ in sol]), np.array([b for _, b in sol]), T)
        filas += evaluar(T, {v: P[:, j] for j, v in enumerate(variantes)}, k)
    return pd.DataFrame(filas)


def exportar(nombre, df, index=True):
    SALIDA.mkdir(parents=True, exist_ok=True)
    ruta = SALIDA / f"{nombre}.csv"
    df.to_csv(ruta, index=index)
    return ruta
