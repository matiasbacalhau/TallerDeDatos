"""Etapa B — interpretación de los dos modelos del producto (pliegues de reporte).

Figuras en figuras/ (PNG para el informe) y tablas en data/resultados/interp_*.csv:
  1. interp_curvas_ridge.png   curvas del modelo interpretable, en °C, con el entrenamiento del último pliegue
  2. interp_shap_lgbm.png      importancia SHAP de LightGBM (|contribución| media en °C)
     interp_latlon.png         ¿latitud y longitud reemplazan al sesgo de cada estación?
  3. interp_sesgo_hora.png     sesgo restante por hora solar: ¿los modelos captaron el error sistemático?
  4. interp_techo.png          correlación entre el error y lo que se sabe al emitir, por horizonte

SHAP: contribuciones exactas de LightGBM (TreeSHAP, predict(pred_contrib=True)).
Uso:  .venv/bin/python src/interpretar.py   (requiere haber corrido modelo_interpretable.py y modelo_lgbm.py)
"""
import json

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import modelo_interpretable as MI
import modelo_lgbm as ML
from modelado import RAIZ, SEMILLA, cargar_entrenamiento, pliegues_eleccion_reporte, preparar_prueba

RES = RAIZ / "data" / "resultados"
FIG = RAIZ / "figuras"
# paleta de referencia (skill dataviz, modo claro): slots categóricos en orden fijo y tintas de texto
SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TXT, TXT2, GRILLA, FONDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
N_SHAP = 100_000
MODELOS = {"B0": "NWS crudo (B0)", "B2": "Estación × hora (B2)", "lineal": "Ridge", "lgbm": "LightGBM"}


def estilo():
    plt.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO, "font.size": 9,
        "font.family": ["Helvetica Neue", "DejaVu Sans"], "text.color": TXT, "axes.labelcolor": TXT2,
        "axes.edgecolor": GRILLA, "xtick.color": TXT2, "ytick.color": TXT2, "axes.grid": True,
        "grid.color": GRILLA, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
        "legend.frameon": False, "axes.axisbelow": True})


def guardar(fig, nombre):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / nombre, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 1. curvas del Ridge
def efecto(dis, coef, prefijo, B):
    """Σ β_i (B_i(x) - media_i) en °C: el aporte de un bloque spline respecto de su promedio."""
    idx = [i for i, n in enumerate(dis.nombres) if n.startswith(prefijo)]
    b = coef[idx] / dis.esc.scale_[idx]
    return (B - dis.esc.mean_[idx]) @ b


def curvas_ridge(dis, m, A):
    coef = m.coef_
    paneles = []
    x = np.linspace(0, 24, 241)
    paneles.append(("Hora solar", "hora solar (h)", x, efecto(dis, coef, "hora_s", dis.sp_hora.transform(x[:, None]))))
    unidades = {"pendiente_pron": ("Pendiente del pronóstico", "°C/h"), "temp_anom": ("Anomalía del pronóstico", "°C"),
                "dep_rocio": ("Depresión del punto de rocío", "°C"), "viento_vel": ("Velocidad del viento", "km/h")}
    for c, (titulo, u) in unidades.items():
        lo, hi = np.nanpercentile(A[c], [1, 99])
        x = np.linspace(lo, hi, 200)
        paneles.append((titulo, f"{c} ({u})", x, efecto(dis, coef, f"{c}_s", dis.sp[c].transform(pd.DataFrame({c: x})))))
    h = np.arange(1, 49)
    G = dis.sp_h.transform(pd.DataFrame({"horizonte": h}))
    i0 = dis.nombres.index("err_ult")
    idx = [i for i, n in enumerate(dis.nombres) if n.startswith("err_ult_x_h_s")]
    peso = coef[i0] / dis.esc.scale_[i0] + G @ (coef[idx] / dis.esc.scale_[idx])

    fig, ax = plt.subplots(2, 3, figsize=(10, 5.6))
    lim = max(np.abs(p[3]).max() for p in paneles) * 1.1
    filas = []
    for a, (titulo, xl, x, y) in zip(ax.flat, paneles):
        a.axhline(0, color=TXT2, lw=0.8)
        a.plot(x, y, color=SERIE[0])
        a.set(title=titulo, xlabel=xl, ylim=(-lim, lim))
        filas += [{"curva": titulo, "x": xi, "efecto_C": yi} for xi, yi in zip(x, y)]
    for a in ax[:, 0]:
        a.set_ylabel("aporte a la corrección (°C)")
    a = ax.flat[5]
    a.axhline(0, color=TXT2, lw=0.8)
    a.plot(h, peso, color=SERIE[1])
    a.set(title="Peso del último error conocido", xlabel="horizonte (h)", ylabel="°C de corrección por °C de err_ult")
    filas += [{"curva": "peso err_ult", "x": hi, "efecto_C": yi} for hi, yi in zip(h, peso)]
    i = dis.nombres.index("enfriamiento_nocturno")
    enfr = coef[i] / dis.esc.scale_[i]
    fig.suptitle(f"Modelo interpretable (Ridge): qué corrige cada variable · enfriamiento nocturno: {enfr:+.2f} °C",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    guardar(fig, "interp_curvas_ridge.png")
    pd.DataFrame(filas).to_csv(RES / "interp_curvas_ridge.csv", index=False)
    return enfr, peso


# ---------------------------------------------------------------- 2. SHAP de LightGBM
def shap_lgbm(gbm, T, A, rng):
    S = T.iloc[rng.choice(len(T), min(N_SHAP, len(T)), replace=False)]
    C = gbm.predict(ML.matriz(S), pred_contrib=True)              # última columna: valor esperado
    C = pd.DataFrame(C[:, :-1], columns=gbm.feature_name(), index=S.index)
    imp = C.abs().mean().sort_values()
    imp.rename("shap_abs_medio_C").to_csv(RES / "interp_shap_lgbm.csv")
    top = imp.tail(15)
    fig, a = plt.subplots(figsize=(6.5, 4.6))
    a.barh(top.index, top.values, color=SERIE[0], height=0.6)
    for y, v in enumerate(top.values):
        a.text(v, y, f" {v:.2f}", va="center", color=TXT2, fontsize=8)
    a.set(xlabel="|contribución SHAP| media (°C)", title="LightGBM: cuánto mueve cada variable la corrección")
    a.grid(axis="y", visible=False)
    guardar(fig, "interp_shap_lgbm.png")

    # latitud y longitud: ¿identifican a la estación? Se compara su aporte medio por estación con el sesgo
    # histórico de la estación en el entrenamiento (error medio con observación real).
    sesgo_est = A[~A["obs_interpolada"]].groupby(A["station_id"].astype(str))["error_c"].mean()
    g = pd.DataFrame({"est": S["station_id"].astype(str).to_numpy(), "latlon": (C["lat"] + C["lon"]).to_numpy(),
                      "sesgos": C[[c for c in C if c.startswith("sesgo_ewma")]].sum(axis=1).to_numpy()})
    g = g.groupby("est").agg(latlon=("latlon", "mean"), sesgos=("sesgos", "mean"), n=("latlon", "size"))
    g = g[g["n"] >= 50].join(sesgo_est.rename("sesgo_hist"), how="inner")
    r_ll, r_ss = g["latlon"].corr(g["sesgo_hist"]), g["sesgos"].corr(g["sesgo_hist"])
    g.to_csv(RES / "interp_latlon.csv")
    fig, a = plt.subplots(figsize=(6.5, 4.2))
    a.axhline(0, color=TXT2, lw=0.8)
    a.axvline(0, color=TXT2, lw=0.8)
    a.scatter(g["sesgo_hist"], g["latlon"], s=14, color=SERIE[0], alpha=0.7, edgecolor=FONDO, linewidth=0.5)
    a.set(xlabel="sesgo histórico de la estación en el entrenamiento (°C)",
          ylabel="aporte medio de lat + lon (°C)",
          title=f"¿Latitud y longitud reemplazan al sesgo de la estación? r = {r_ll:.2f} ({len(g)} estaciones)")
    guardar(fig, "interp_latlon.png")
    return imp, r_ll, r_ss, C


# ---------------------------------------------------------------- 3 y 4. sesgo restante y techo
class Acumulador:
    """Sumas por hora solar y por estación (sesgo restante) y por horizonte (correlaciones)."""

    def __init__(self):
        self.hora, self.est, self.corr = [], [], []

    def agregar(self, T, pred):
        e = T["error_c"].to_numpy()
        R = pd.DataFrame({f"r_{n}": e - p for n, p in pred.items()})
        R["hora"], R["est"] = T["hora"].to_numpy(), T["sid"].to_numpy()
        self.hora.append(R.drop(columns="est").groupby("hora").agg(["sum", "count"]))
        self.est.append(R.drop(columns="hora").groupby("est").agg(["sum", "count"]))
        for nombre, x in (("err_ult", T["err_ult"].to_numpy()), ("Ridge", pred["lineal"]), ("LightGBM", pred["lgbm"])):
            ok = ~np.isnan(x)
            d = pd.DataFrame({"h": T["horizonte"].to_numpy()[ok], "x": x[ok], "y": e[ok]})
            d = d.assign(xx=d.x ** 2, yy=d.y ** 2, xy=d.x * d.y).groupby("h").agg(["sum", "count"])
            self.corr.append((nombre, d))

    @staticmethod
    def medias(partes):
        s = pd.concat(partes).groupby(level=0).sum()
        return pd.DataFrame({c[0][2:]: s[(c[0], "sum")] / s[(c[0], "count")] for c in s.columns if c[1] == "sum"})

    def correlaciones(self):
        out = {}
        for fuente in ("err_ult", "Ridge", "LightGBM"):
            s = pd.concat([d for f, d in self.corr if f == fuente]).groupby(level=0).sum()
            n = s[("x", "count")]
            sx, sy, sxx, syy, sxy = (s[(c, "sum")] for c in ("x", "y", "xx", "yy", "xy"))
            out[fuente] = (n * sxy - sx * sy) / np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
        return pd.DataFrame(out)


def figura_sesgo_hora(M):
    fig, a = plt.subplots(figsize=(7, 4))
    a.axhline(0, color=TXT2, lw=0.8)
    for i, (n, etiqueta) in enumerate(MODELOS.items()):
        a.plot(M.index, M[n], color=SERIE[i], label=etiqueta, marker="o", markersize=3)
    a.set(xlabel="hora solar", ylabel="error medio restante (°C, observado − corregido)", xticks=range(0, 24, 3),
          xlim=(-0.5, 23.5), title="Sesgo restante por hora del día (pliegues 3–4)")
    a.legend(loc="upper left", ncols=4, fontsize=8, bbox_to_anchor=(0, -0.15))
    guardar(fig, "interp_sesgo_hora.png")


def figura_techo(Cr):
    fig, a = plt.subplots(figsize=(7, 4))
    a.axhline(0, color=TXT2, lw=0.8)
    nombres = {"err_ult": "Último error conocido (err_ult)", "Ridge": "Predicción del Ridge", "LightGBM": "Predicción de LightGBM"}
    for i, (c, etiqueta) in enumerate(nombres.items()):
        a.plot(Cr.index, Cr[c], color=SERIE[i], label=etiqueta)
    a.set(xlabel="horizonte (h)", ylabel="correlación con el error del NWS", ylim=(min(-0.15, Cr.min().min() * 1.1), max(0.8, Cr.max().max() * 1.1)),
          xlim=(1, 48), title="Cuánto del error se puede anticipar al emitir el pronóstico")
    a.legend(loc="upper right", fontsize=8)
    guardar(fig, "interp_techo.png")


def main():
    estilo()
    rng = np.random.default_rng(SEMILLA)
    _, reporte = pliegues_eleccion_reporte()
    hp_lin = json.loads((RES / "interpretable_hiperparametros.json").read_text())
    columnas = list(dict.fromkeys(MI.COLUMNAS + ML.COLUMNAS))
    acc = Acumulador()
    for j, (k, (tr, te)) in enumerate(reporte):
        A = cargar_entrenamiento(tr, columnas)
        T = preparar_prueba(te, tr, columnas)
        T = T[T["B2"].notna()].reset_index(drop=True)
        dis, m = MI.ajustar(A, hp_lin["sesgo"])
        gbm = lgb.Booster(model_file=str(RES / f"lgbm_pliegue_{k}.txt"))
        pred = {"B0": np.zeros(len(T)), "B2": T["B2"].to_numpy(), "lineal": m.predict(dis.transform(T)),
                "lgbm": gbm.predict(ML.matriz(T))}
        acc.agregar(T, pred)
        if j == len(reporte) - 1:                      # el último pliegue tiene el entrenamiento más largo
            enfr, peso = curvas_ridge(dis, m, A)
            imp, r_ll, r_ss, _ = shap_lgbm(gbm, T, A, rng)
        print(f"pliegue {k}: {len(T):,} filas de prueba", flush=True)
        del A, T

    H = acc.medias(acc.hora)
    H.to_csv(RES / "interp_sesgo_hora.csv")
    figura_sesgo_hora(H)
    E = acc.medias(acc.est)
    Cr = acc.correlaciones()
    Cr.to_csv(RES / "interp_techo.csv")
    figura_techo(Cr)

    pd.set_option("display.width", 200)
    print(f"\n1. Ridge: enfriamiento nocturno {enfr:+.2f} °C; peso de err_ult a 1/6/24/48 h: "
          f"{peso[0]:.2f} / {peso[5]:.2f} / {peso[23]:.2f} / {peso[47]:.2f}")
    print("\n2. SHAP (|contribución| media, °C), 10 mayores:")
    print(imp.sort_values(ascending=False).head(10).round(3).to_string())
    print(f"   correlación por estación con su sesgo histórico: lat+lon r = {r_ll:.2f} | sesgos exponenciales r = {r_ss:.2f}")
    print("\n3. sesgo restante: amplitud del ciclo por hora (máx − mín) y dispersión entre estaciones (desvío de la media por estación), °C")
    print(pd.DataFrame({"amplitud_hora": H.max() - H.min(), "desvío_entre_estaciones": E.std(),
                        "media_de_las_horas": H.mean()}).round(3).to_string())
    print("\n4. correlación con el error por horizonte:")
    print(Cr.loc[[1, 3, 6, 12, 24, 36, 48]].round(3).to_string())


if __name__ == "__main__":
    main()
