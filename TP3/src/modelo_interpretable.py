"""Modelo de caja de cristal: regresión lineal regularizada (Ridge) con curvas, para cualquier lista de familias.

  corrección = b0 + Σ β·x (columnas lineales, con indicador de faltante) + f(hora solar) + f(pendiente_pron)
               + f(temp_anom) + f(dep_rocio) + f(viento_vel) + err_ult · g(horizonte) + vec_err_1h · g'(horizonte)
               + enfriamiento nocturno + interacciones (sesgo exponencial × noche, err_ayer × condición)

Cada f y g es una combinación de bases spline, así que se lee como una curva. La hora solar usa un spline periódico
(las 23 h y las 0 h son vecinas). Qué entra depende de las familias pedidas (`Disenio(familias)`): cada columna de la
matriz sabe a qué familia pertenece (`familia_de`), y así se mide cuánto aporta cada familia y se la puede sacar.

Qué familias usa el producto lo decide la experimentación (notebooks/experimentos.ipynb, regla de recorte en
DECISIONES.md) y queda en config.yaml (modelos.interpretable.familias). LightGBM, en cambio, usa siempre las 8.

Columnas por familia:
  base        prob_lluvia, horizonte, condición (8 indicadores), enfriamiento nocturno = noche × despejado × viento
              < umbral_calma_kmh, spline de hora solar, curvas de temp_anom, dep_rocio y viento_vel y, como lineales,
              el resto de la familia (temperatura, humedad, rocío, dirección del viento, elevación del sol, lat, lon)
  pendiente   curva de pendiente_pron
  rezagos     sus 4 columnas, err_ult × g(horizonte) y err_ayer × condición (la persistencia depende del tiempo)
  sesgo_ewma  los 4 sesgos y sesgo × noche (el sesgo de la estación pesa distinto de noche)
  vecinos     sus 2 columnas y vec_err_1h × g'(horizonte)
  revision, rezagos_extra, dinamica: sus columnas, como lineales
  Los indicadores de faltante van con su variable. La base spline del horizonte que pondera err_ult existe aunque
  se saque la familia base.

Pasos (main):
  1. En los pliegues de elección se elige la penalización α (una sola vez; queda congelada en
     interpretable_hiperparametros.json). Se vuelve a elegir si cambian las familias o con --reelegir.
  2. Con ese α se entrena y se evalúa cada pliegue de reporte, contra B0 y B2, por segmento. Cada pliegue queda
     en la caché (data/cache/ridge/): si sus datos no cambiaron, no se vuelve a entrenar.

Uso:  .venv/bin/python src/modelo_interpretable.py [--reelegir]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler

import cache
from formato import dec, n as n_
from modelado import CFG, RAIZ, SEMILLA, cargar_entrenamiento, dias_historia_incompleta, evaluar, pliegues_eleccion_reporte, preparar_prueba
from ridge_gram import Gram

P = CFG["modelos"]["interpretable"]
VERSION = 2                                   # cambia si cambia la forma de la matriz (invalida la caché)
FAMILIAS = json.loads((RAIZ / "data" / "features" / "familias.json").read_text())
TODAS = list(FAMILIAS)
FAMILIA_DE = {c: f for f, cols in FAMILIAS.items() for c in cols}
CONDICIONES = ["despejado", "parcial", "nublado", "lluvia", "tormenta", "niebla", "nieve", "otro"]
LINEALES = ["prob_lluvia", "horizonte"]                         # faltante -> mediana del entrenamiento
CURVAS = ["pendiente_pron", "temp_anom", "dep_rocio", "viento_vel"]
APARTE = set(LINEALES + CURVAS + ["hora_solar_sin", "hora_solar_cos", "noche", "condicion"])   # tienen forma propia
SIEMPRE = ["station_id", "target_time_utc", "error_c", "obs_interpolada", "horizonte", "hora_solar_sin",
           "hora_solar_cos", "noche", "condicion"]               # evaluación, B2, interacciones y spline del horizonte
POR_HORIZONTE = {"err_ult": "rezagos", "vec_err_1h": "vecinos"}  # su peso depende de la antelación
BLOQUE = 200_000                                                 # filas por bloque al predecir (memoria)
SESGO = "los 4 (lineales)"      # antes se elegía UN sesgo exponencial; ahora entran los 4 como columnas


def alphas():
    return sorted(float(a) for a in P["ridge_alphas"])          # YAML lee 1.0e4 como texto


def columnas_de(familias):
    """Columnas de data/features que hay que leer para estas familias."""
    return list(dict.fromkeys(SIEMPRE + ["viento_vel"] + [c for f in familias for c in FAMILIAS[f]]))


class Disenio:
    """Matriz de diseño. Splines, medianas y escalado se ajustan solo con el entrenamiento del pliegue."""

    def __init__(self, familias):
        self.familias = [f for f in TODAS if f in set(familias)]
        f = set(self.familias)
        self.base = "base" in f
        self.lineales = LINEALES if self.base else []
        self.curvas = [c for c in CURVAS if FAMILIA_DE[c] in f]
        self.con_faltante = [c for fam in self.familias for c in FAMILIAS[fam] if c not in APARTE]
        self.por_horizonte = [c for c, fam in POR_HORIZONTE.items() if fam in f]
        self.sesgos = FAMILIAS["sesgo_ewma"] if "sesgo_ewma" in f else []
        self.ayer_x_cond = "rezagos" in f

    def fit(self, df):
        med = self.lineales + [c for c in ("dep_rocio", "viento_vel", "pendiente_pron") if c in self.curvas]
        self.medianas = df[med].median()
        df = self._llenar(df)
        if self.base:
            self.sp_hora = SplineTransformer(n_knots=7, degree=3, extrapolation="periodic").fit(self._hora(df))
        self.sp = {c: SplineTransformer(n_knots=6, degree=3, extrapolation="linear").fit(df[[c]].dropna())
                   for c in self.curvas}               # nudos con los valores observados (temp_anom tiene faltantes)
        self.sp_h = SplineTransformer(n_knots=5, degree=2, extrapolation="constant").fit(df[["horizonte"]])
        X, self.nombres, self.familia_de = self._crudo(df)
        self.esc = StandardScaler().fit(X)
        return self

    def transform(self, df):
        X, _, _ = self._crudo(self._llenar(df))
        return self.esc.transform(X).astype(np.float32)

    def _llenar(self, df):
        """Faltantes: lineales con indicador y temp_anom -> 0 (en _crudo); prob_lluvia, horizonte y curvas -> mediana."""
        df = df.copy()
        df[self.medianas.index] = df[self.medianas.index].fillna(self.medianas)
        return df

    @staticmethod
    def _hora(df):
        ang = np.arctan2(df["hora_solar_sin"].to_numpy(), df["hora_solar_cos"].to_numpy())
        return ((ang / (2 * np.pi) * 24) % 24).reshape(-1, 1)

    def _crudo(self, df):
        bloques, nombres, familia = [], [], []

        def agregar(M, ns, fam):
            bloques.append(np.asarray(M, dtype=np.float64).reshape(len(df), -1))
            nombres.extend(ns)
            familia.extend([fam] * len(ns))

        cond = df["condicion"].astype(str).to_numpy()
        if self.base:
            agregar(df[self.lineales].to_numpy(), self.lineales, "base")
            agregar(np.column_stack([cond == c for c in CONDICIONES]), [f"cond_{c}" for c in CONDICIONES], "base")
            enfr = df["noche"].to_numpy(bool) & (cond == "despejado") & (df["viento_vel"].to_numpy() < P["umbral_calma_kmh"])
            agregar(enfr, ["enfriamiento_nocturno"], "base")
            H = self.sp_hora.transform(self._hora(df))
            agregar(H, [f"hora_s{i}" for i in range(H.shape[1])], "base")
        for c in self.con_faltante:                     # faltante -> 0 + indicador
            v = df[c].to_numpy(dtype=np.float64)
            falta = np.isnan(v)
            agregar(np.where(falta, 0.0, v), [c], FAMILIA_DE[c])
            agregar(falta, [f"falta_{c}"], FAMILIA_DE[c])
        for c, sp in self.sp.items():
            v = df[[c]].to_numpy(dtype=np.float64)
            if c == "temp_anom":                        # sin semana previa: anomalía 0 + indicador
                falta = np.isnan(v[:, 0])
                agregar(falta, ["falta_temp_anom"], "base")
                v = np.where(np.isnan(v), 0.0, v)
            S = sp.transform(pd.DataFrame(v, columns=[c]))
            agregar(S, [f"{c}_s{i}" for i in range(S.shape[1])], FAMILIA_DE[c])
        if self.por_horizonte:
            G = self.sp_h.transform(df[["horizonte"]])
            for c in self.por_horizonte:
                e = np.nan_to_num(df[c].to_numpy(dtype=np.float64))
                agregar(G * e[:, None], [f"{c}_x_h_s{i}" for i in range(G.shape[1])], POR_HORIZONTE[c])
        noche = df["noche"].to_numpy(np.float64)
        for c in self.sesgos:                           # el sesgo de la estación pesa distinto de noche
            agregar(np.nan_to_num(df[c].to_numpy(np.float64)) * noche, [f"{c}_x_noche"], "sesgo_ewma")
        if self.ayer_x_cond:                            # persistencia según el tiempo
            ayer = np.nan_to_num(df["err_ayer"].to_numpy(np.float64))
            agregar(np.column_stack([ayer * (cond == c) for c in CONDICIONES]),
                    [f"err_ayer_x_{c}" for c in CONDICIONES], "rezagos")
        return np.hstack(bloques), nombres, familia


def ajustar(A, familias, alpha):
    dis = Disenio(familias).fit(A)
    m = Ridge(alpha=alpha).fit(dis.transform(A), A["error_c"].to_numpy())
    return dis, m


def predecir(dis, coef, intercepto, T):
    """Predicción por bloques de filas (la matriz completa de prueba no entra en memoria). coef: (p,) o (p, m)."""
    partes = [dis.transform(T.iloc[i:i + BLOQUE]) @ coef + intercepto for i in range(0, len(T), BLOQUE)]
    return np.concatenate(partes).astype(np.float32)


def contribuciones(dis, coef, X):
    """Aporte de cada familia a la predicción, fila por fila (Σ β·x de sus columnas, escala estandarizada)."""
    fam = np.array(dis.familia_de)
    return pd.DataFrame({f: X[:, fam == f] @ coef[fam == f] for f in dis.familias})


def contribucion_por_familia(dis, coef, X):
    """Desvío estándar del aporte de cada familia (°C): cuánto mueve la predicción."""
    return contribuciones(dis, coef, X).std()


def hiperparametros_vigentes(ruta, familias):
    """Los hiperparámetros guardados, o None si no existen o son de otro conjunto de familias."""
    if not Path(ruta).exists():
        return None
    hp = json.loads(Path(ruta).read_text())
    return hp if hp.get("familias") == list(familias) else None


def mae_completa(T, preds):
    """MAE de cada columna de preds (n, m) en las filas con historia completa y B2 definido (criterio de elección)."""
    ok = (~T["dia"].isin(set(dias_historia_incompleta())) & T["B2"].notna()).to_numpy()
    return np.abs(T["error_c"].to_numpy()[ok, None] - preds[ok]).mean(axis=0)


def elegir_alpha(familias, eleccion, max_filas):
    """α con menor MAE medio (historia completa) en los pliegues de elección. Si queda en un borde de la grilla,
    se agrega un orden de magnitud hacia ese lado (hasta dos veces).
    Devuelve (grilla: MAE por pliegue y α; α elegido; filas de evaluar() con ese α, por pliegue y segmento)."""
    grilla = alphas()
    datos = []
    for k, (tr, te) in eleccion:
        A = cargar_entrenamiento(tr, columnas_de(familias), max_filas)
        dis = Disenio(familias).fit(A)
        g = Gram.desde(dis.transform(A), A["error_c"].to_numpy())
        datos.append((k, dis, g, preparar_prueba(te, tr, columnas_de(familias))))
        del A
    filas = []
    for _ in range(3):
        nuevos = [a for a in grilla if a not in {f["alpha"] for f in filas}]
        for k, dis, g, T in datos:
            sol = [g.resolver(np.ones(len(dis.nombres), bool), a) for a in nuevos]
            pred = predecir(dis, np.column_stack([c for c, _ in sol]), np.array([b for _, b in sol]), T)
            filas += [{"pliegue": k, "alpha": a, "MAE": m} for a, m in zip(nuevos, mae_completa(T, pred))]
        G = pd.DataFrame(filas)
        crit = G.groupby("alpha")["MAE"].mean()
        mejor = crit.idxmin()
        if mejor == min(grilla):
            grilla.append(mejor / 10)
        elif mejor == max(grilla):
            grilla.append(mejor * 10)
        else:
            break
    sel = []                                     # con el α elegido: filas de evaluar() (formato que leen las notebooks)
    for k, dis, g, T in datos:
        coef, b0 = g.resolver(np.ones(len(dis.nombres), bool), mejor)
        sel += [f for f in evaluar(T, {"lineal": predecir(dis, coef, b0, T)}, k)
                if f["modelo"] == "lineal" and f["horizonte"] == "todos"]
    return G, float(mejor), pd.DataFrame(sel).assign(alpha=float(mejor), sesgo=SESGO)


def clave_pliegue(tipo, familias, alpha, tr, te, max_filas):
    """Todo lo que determina el resultado de un pliegue: si nada cambió, se lee de la caché."""
    return cache.clave(tipo=tipo, version=VERSION, familias=list(familias), alpha=alpha, max_filas=max_filas,
                       semilla=SEMILLA, umbral_calma=P["umbral_calma_kmh"], entrena=[f"{d:%Y_%m_%d}" for d in tr],
                       prueba=[f"{d:%Y_%m_%d}" for d in te], huellas=cache.huella_dias(list(tr) + list(te)),
                       historia_incompleta=sorted(dias_historia_incompleta()))


def pliegue_reporte(k, tr, te, familias, alpha, max_filas):
    """Filas de evaluar() y coeficientes de un pliegue de reporte (de la caché o entrenando)."""
    clave = clave_pliegue("ridge", familias, alpha, tr, te, max_filas)
    res = cache.leer("ridge", clave)
    if res is not None:
        return res, True
    A = cargar_entrenamiento(tr, columnas_de(familias), max_filas)
    T = preparar_prueba(te, tr, columnas_de(familias))
    dis, m = ajustar(A, familias, alpha)
    res = {"filas": evaluar(T, {"lineal": predecir(dis, m.coef_, m.intercept_, T)}, k),
           "coef": pd.Series(m.coef_, index=dis.nombres, name=f"pliegue_{k}"),
           "familia_de": dict(zip(dis.nombres, dis.familia_de)), "filas_entrenamiento": len(A)}
    cache.guardar("ridge", clave, res)
    return res, False


def main():
    salida = RAIZ / "data" / "resultados"
    salida.mkdir(exist_ok=True)
    familias, max_filas = list(P["familias"]), P["max_filas"]
    ruta_hp = salida / "interpretable_hiperparametros.json"
    eleccion, reporte = pliegues_eleccion_reporte()
    print("Modelo de caja de cristal (Ridge con curvas). Familias que usa:", ", ".join(familias))
    print(f"Pliegues que eligen la penalización: {[k for k, _ in eleccion]} · pliegues que reportan: {[k for k, _ in reporte]}")

    hp = None if "--reelegir" in sys.argv else hiperparametros_vigentes(ruta_hp, familias)
    if hp is None:
        print("\nEligiendo la penalización α en los pliegues de elección (criterio: MAE con historia completa)...", flush=True)
        G, alpha, sel = elegir_alpha(familias, eleccion, max_filas)
        G.to_csv(salida / "interpretable_grilla_alpha.csv", index=False)
        sel.to_csv(salida / "interpretable_eleccion.csv", index=False)
        print("  α          MAE medio (°C)")
        for a_, m_ in G.groupby("alpha")["MAE"].mean().items():
            print(f"  {a_:<10.0e} {dec(m_, 4)}" + ("   ← elegido" if a_ == alpha else ""))
        ruta_hp.write_text(json.dumps({**P, "familias": familias, "ridge_alpha": alpha, "sesgo": SESGO,
                                       "pliegues_eleccion": [k for k, _ in eleccion]}))
    else:
        alpha = hp["ridge_alpha"]
        print(f"\nPenalización ya elegida (congelada en {ruta_hp.relative_to(RAIZ)}): α = {alpha:.0e}. "
              "Para volver a elegirla: --reelegir")

    filas, coefs, fam_de = [], [], {}
    for k, (tr, te) in reporte:
        t0 = time.time()
        res, de_cache = pliegue_reporte(k, tr, te, familias, alpha, max_filas)
        filas += res["filas"]
        coefs.append(res["coef"])
        fam_de.update(res["familia_de"])
        print(f"pliegue {k}: " + ("leído de la caché" if de_cache else
              f"entrenado con {n_(res['filas_entrenamiento'])} filas en {time.time() - t0:.0f} s"), flush=True)
    R = pd.DataFrame(filas)
    R.to_csv(salida / "interpretable_por_pliegue.csv", index=False)
    pd.concat(coefs, axis=1).to_csv(salida / "interpretable_coeficientes.csv")      # solo números (lo leen notebooks)
    pd.Series(fam_de, name="familia").rename_axis("columna").to_csv(salida / "interpretable_familia_de_columna.csv")

    print("\nMAE en °C de la temperatura corregida (más bajo es mejor), promedio de los pliegues de reporte.")
    print("B0 = NWS sin corregir; B2 = error medio de la estación a esa hora; lineal = este modelo.")
    for seg in ("completa", "incompleta", "todas"):
        x = R[R.segmento == seg]
        if x.empty:
            continue
        print(f"\nDías con historia {seg} · pliegues {sorted(int(v) for v in x.pliegue.unique())}"
              if seg != "todas" else f"\nTodos los días de prueba · pliegues {sorted(int(v) for v in x.pliegue.unique())}")
        t = x.astype({"horizonte": str}).groupby(["horizonte", "modelo"])["MAE"].mean().unstack()
        print(f"  {'horizonte':<10}" + "".join(f"{c:>9}" for c in ("B0", "B2", "lineal")))
        for h in ["todos", "1", "3", "6", "12", "24", "48"]:
            print(f"  {h + ('' if h == 'todos' else ' h'):<10}" + "".join(f"{dec(t.loc[h, c], 3):>9}" for c in ("B0", "B2", "lineal")))
    print("\nResultados: data/resultados/interpretable_por_pliegue.csv, interpretable_coeficientes.csv, "
          "interpretable_familia_de_columna.csv")


if __name__ == "__main__":
    main()
