"""Modelo que prioriza la interpretabilidad: regresión lineal (Ridge) con curvas en pocas variables.

  corrección = b0 + Σ β·x (variables lineales) + f(hora solar) + f(pendiente_pron) + f(temp_anom)
               + f(dep_rocio) + f(viento_vel) + err_ult · g(horizonte) + γ · enfriamiento_nocturno

Cada f y g es una combinación de bases spline, así que se lee como una curva. La hora solar usa un spline
periódico (las 23 h y las 0 h son vecinas).

Variables elegidas por criterio meteorológico (DECISIONES.md, Etapa A — Modelos):
  - sesgo exponencial por estación × hora (UN solo α, elegido en los pliegues de elección): error propio de
    la estación (altura, entorno, valle, costa).
  - rezagos: persistencia de la situación; el último error pesa según la antelación -> err_ult · g(horizonte).
  - hora solar: ciclo diario y transiciones de la mañana y la tarde.
  - pendiente del pronóstico: errores de timing de frentes y del ciclo diario, desfase de la observación.
  - anomalía de la temperatura pronosticada: los pronósticos suavizan los extremos (en valor absoluto la
    curva aprendería la estación del año).
  - depresión del rocío: cerca de la saturación la temperatura casi no puede bajar del punto de rocío.
  - velocidad del viento (con umbral): mezcla de la capa cercana al suelo.
  - condición y probabilidad de lluvia: nubosidad y enfriamiento por evaporación.
  - enfriamiento nocturno = noche (sol bajo el horizonte) × despejado × viento < umbral_calma_kmh: inversión
    por radiación, que los modelos subestiman.
Quedan fuera (solo en LightGBM): humedad relativa y punto de rocío (los representa la depresión del rocío),
dirección del viento (efecto local; diagnóstico en src/diagnostico_viento.py), latitud y longitud.

Pasos:
  1. En la primera mitad de los pliegues se elige el α del sesgo exponencial (único hiperparámetro).
  2. Con ese α fijo se reentrena y se evalúa en la segunda mitad, contra B0 y B2, por segmento.
  3. Se guardan los coeficientes por pliegue.

Uso:  .venv/bin/python src/modelo_interpretable.py
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler

from modelado import CFG, RAIZ, cargar_entrenamiento, evaluar, pliegues_eleccion_reporte, preparar_prueba

P = CFG["modelos"]["interpretable"]
LINEALES = ["prob_lluvia", "horizonte"]
CURVAS = ["pendiente_pron", "temp_anom", "dep_rocio", "viento_vel"]
REZAGOS = ["err_ult", "err_media3", "err_media6", "err_ayer"]
CONDICIONES = ["despejado", "parcial", "nublado", "lluvia", "tormenta", "niebla", "nieve", "otro"]
SESGOS = [f"sesgo_ewma_{int(round(a * 100)):02d}" for a in CFG["features"]["alfas_ewma"]]
COLUMNAS = (["station_id", "target_time_utc", "error_c", "obs_interpolada", "hora_solar_sin", "hora_solar_cos",
             "noche", "condicion"] + LINEALES + CURVAS + REZAGOS + SESGOS)


class Disenio:
    """Matriz de diseño. Splines, medianas y escalado se ajustan solo con el entrenamiento del pliegue."""

    def __init__(self, sesgo):
        self.sesgo = sesgo

    def fit(self, df):
        self.medianas = df[LINEALES + ["dep_rocio", "viento_vel", "pendiente_pron"]].median()
        df = self._llenar(df)
        self.sp_hora = SplineTransformer(n_knots=7, degree=3, extrapolation="periodic").fit(self._hora(df))
        self.sp = {c: SplineTransformer(n_knots=6, degree=3, extrapolation="linear").fit(df[[c]].dropna())
                   for c in CURVAS}                     # nudos con los valores observados (temp_anom tiene faltantes)
        self.sp_h = SplineTransformer(n_knots=5, degree=2, extrapolation="constant").fit(df[["horizonte"]])
        X, self.nombres = self._crudo(df)
        self.esc = StandardScaler().fit(X)
        return self

    def transform(self, df):
        X, _ = self._crudo(self._llenar(df))
        return self.esc.transform(X).astype(np.float32)

    def _llenar(self, df):
        """Faltantes: rezagos, sesgo y anomalía -> 0 (con indicador, en _crudo); el resto -> mediana."""
        df = df.copy()
        df[self.medianas.index] = df[self.medianas.index].fillna(self.medianas)
        return df

    @staticmethod
    def _hora(df):
        ang = np.arctan2(df["hora_solar_sin"].to_numpy(), df["hora_solar_cos"].to_numpy())
        return ((ang / (2 * np.pi) * 24) % 24).reshape(-1, 1)

    def _crudo(self, df):
        bloques, nombres = [], []

        def agregar(M, ns):
            bloques.append(np.asarray(M, dtype=np.float64).reshape(len(df), -1))
            nombres.extend(ns)

        agregar(df[LINEALES].to_numpy(), LINEALES)
        cond = df["condicion"].astype(str).to_numpy()
        agregar(np.column_stack([cond == c for c in CONDICIONES]), [f"cond_{c}" for c in CONDICIONES])
        enfr = df["noche"].to_numpy(bool) & (cond == "despejado") & (df["viento_vel"].to_numpy() < P["umbral_calma_kmh"])
        agregar(enfr, ["enfriamiento_nocturno"])
        for c in REZAGOS + [self.sesgo]:                # faltante -> 0 + indicador
            v = df[c].to_numpy(dtype=np.float64)
            falta = np.isnan(v)
            agregar(np.where(falta, 0.0, v), [c])
            agregar(falta, [f"falta_{c}"])
        H = self.sp_hora.transform(self._hora(df))
        agregar(H, [f"hora_s{i}" for i in range(H.shape[1])])
        for c, sp in self.sp.items():
            v = df[[c]].to_numpy(dtype=np.float64)
            if c == "temp_anom":                        # sin semana previa: anomalía 0 + indicador
                falta = np.isnan(v[:, 0])
                agregar(falta, ["falta_temp_anom"])
                v = np.where(np.isnan(v), 0.0, v)
            S = sp.transform(pd.DataFrame(v, columns=[c]))
            agregar(S, [f"{c}_s{i}" for i in range(S.shape[1])])
        G = self.sp_h.transform(df[["horizonte"]])
        e = np.nan_to_num(df["err_ult"].to_numpy(dtype=np.float64))
        agregar(G * e[:, None], [f"err_ult_x_h_s{i}" for i in range(G.shape[1])])
        return np.hstack(bloques), nombres


def ajustar(A, sesgo):
    dis = Disenio(sesgo).fit(A)
    m = Ridge(alpha=P["ridge_alpha"]).fit(dis.transform(A), A["error_c"].to_numpy())
    return dis, m


def main():
    salida = RAIZ / "data" / "resultados"
    salida.mkdir(exist_ok=True)
    eleccion, reporte = pliegues_eleccion_reporte()
    print("eligen:", [k for k, _ in eleccion], "| reportan:", [k for k, _ in reporte])

    # --- 1. elección del α del sesgo exponencial -------------------------------------------
    grilla = []
    for k, (tr, te) in eleccion:
        t0 = time.time()
        A = cargar_entrenamiento(tr, COLUMNAS)
        T = preparar_prueba(te, tr, COLUMNAS)
        for sesgo in SESGOS:
            dis, m = ajustar(A, sesgo)
            for f in evaluar(T, {"lineal": m.predict(dis.transform(T))}, k):
                if f["modelo"] == "lineal" and f["horizonte"] == "todos":
                    grilla.append({**f, "sesgo": sesgo})
        print(f"pliegue {k}: {len(A):,} filas de entrenamiento | {len(T):,} de prueba | {time.time() - t0:.0f} s")
    G = pd.DataFrame(grilla)
    G.to_csv(salida / "interpretable_eleccion.csv", index=False)
    crit = G[G.segmento == "completa"].groupby("sesgo")["MAE"].mean().sort_values()
    print("\nMAE medio (historia completa) en los pliegues de elección:")
    print(crit.round(4).to_string())
    sesgo = crit.index[0]
    print("elegido:", sesgo)

    # --- 2. reporte en la segunda mitad ------------------------------------------------------
    filas, coefs = [], []
    for k, (tr, te) in reporte:
        A = cargar_entrenamiento(tr, COLUMNAS)
        T = preparar_prueba(te, tr, COLUMNAS)
        dis, m = ajustar(A, sesgo)
        filas += evaluar(T, {"lineal": m.predict(dis.transform(T))}, k)
        coefs.append(pd.Series(m.coef_, index=dis.nombres, name=f"pliegue_{k}"))
    R = pd.DataFrame(filas)
    R.to_csv(salida / "interpretable_por_pliegue.csv", index=False)
    C = pd.concat(coefs, axis=1)
    C.to_csv(salida / "interpretable_coeficientes.csv")
    (salida / "interpretable_hiperparametros.json").write_text(json.dumps({"sesgo": sesgo, **P}))

    pd.set_option("display.width", 200)
    for seg in ("completa", "incompleta", "todas"):
        x = R[R.segmento == seg]
        if x.empty:
            continue
        print(f"\n== MAE · segmento {seg} · pliegues {sorted(int(v) for v in x.pliegue.unique())}")
        print(x.groupby(["horizonte", "modelo"])["MAE"].mean().unstack().reindex(["todos", 1, 3, 6, 12, 24, 48]).round(3))
    print("\ncoeficientes más grandes (|β| medio, escala estandarizada):")
    print(C.abs().mean(axis=1).sort_values(ascending=False).head(15).round(3).to_string())
    print("\nenfriamiento_nocturno (β estandarizado por pliegue):", C.loc["enfriamiento_nocturno"].round(3).to_dict())


if __name__ == "__main__":
    main()
