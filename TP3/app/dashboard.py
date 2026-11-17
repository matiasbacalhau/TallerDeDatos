"""Dashboard del TP3 (Etapa C): confianza del pronóstico de temperatura del NWS y comparación de las dos correcciones.

Pestañas:
  0. Hallazgos            la narrativa: qué aprendimos y cuánto confiar en el pronóstico
  1. Resumen              ¿cuál corrige mejor y la diferencia es real? (pliegues de reporte 3–5)
  2. Ridge (Gram)         versiones, familias de variables, búsqueda de α y regularización del modelo de caja de cristal
  3. LightGBM             versiones, variables, grilla de hiperparámetros y regularización del modelo de mínimo error
  4. Ridge vs LightGBM    cara a cara con filtros: dónde gana cada uno, sesgo que queda, banda de confianza
  5. Monitoreo            pronóstico contra pronóstico corregido, con filtros (la vista original de la Etapa C)
  6. Variables            exploración de las 8 familias: correlación con el error y redundancia
  7. Datos y validación   días excluidos, pliegues walk-forward y baselines B0–B3

Datos: app/datos/ (filas de prueba fuera de muestra, de src/preparar_dashboard.py) y app/datos/resultados/ (tablas
de búsqueda y pruebas, de src/preparar_resultados_dashboard.py), más las figuras versionadas de figuras/ y
salidas/experimentos/. El dashboard corre sin los datos crudos ni el resto del pipeline.
Estilo: A, «libro contable acromático» (taste-skill). Cada modelo tiene un color fijo en todas las pestañas:
Ridge azul y LightGBM ámbar; el NWS y los baselines, gris.

Uso (desde TP3/):  streamlit run app/dashboard.py   (instalación: ver README, «Ver el dashboard»)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
DATOS = Path(__file__).resolve().parent / "datos"
RES = DATOS / "resultados"
FIG, FIG_EXP = RAIZ / "figuras", RAIZ / "salidas" / "experimentos" / "figuras"
EXP = RAIZ / "salidas" / "experimentos"
T0 = pd.Timestamp("2026-01-01", tz="UTC")       # origen de horas_desde_t0 (src/preparar_dashboard.py)

# tokens del estilo A; un color fijo por modelo (validado: separación CVD ΔE 26,8 entre azul y ámbar)
TINTA, GRIS, BORDE, REGLA, FONDO, BLANCO = "#0a0a0a", "#6b6b6b", "#e6e6e6", "#f0f0f0", "#f2f4f7", "#ffffff"
AZUL, AMBAR, POSITIVO, NEGATIVO, NWS, B2C = "#1d5fbf", "#c56a00", "#2e7d32", "#c4242a", "#8a8a8a", "#b9b9b9"
BANDA = {"lineal": "rgba(29,95,191,0.14)", "lgbm": "rgba(197,106,0,0.16)"}
COLOR = {"B0": NWS, "nws": NWS, "B2": B2C, "lineal": AZUL, "lgbm": AMBAR}
NOMBRE = {"B0": "NWS crudo (B0)", "nws": "NWS crudo (B0)", "B2": "Estación × hora (B2)", "lineal": "Ridge",
          "lgbm": "LightGBM"}
MODELOS = {"lineal": "Ridge (interpretable)", "lgbm": "LightGBM (mínimo error)"}
CONDICIONES = {"despejado": "Despejado", "parcial": "Parcialmente nublado", "nublado": "Nublado", "lluvia": "Lluvia",
               "tormenta": "Tormenta", "niebla": "Niebla", "nieve": "Nieve", "otro": "Otro"}
FAMILIAS = {   # DECISIONES.md, Etapa A — Feature engineering
    "base": ("El pronóstico mismo y el contexto", "temp_pron, temp_anom, elev_solar, noche, hum_pron, rocio_pron, "
             "dep_rocio, viento_vel, viento_sin/cos, viento_calmo, prob_lluvia, condicion, horizonte, "
             "hora_solar_sin/cos, lat, lon"),
    "pendiente": ("Cuánto cambia la temperatura pronosticada alrededor de la hora objetivo", "pendiente_pron"),
    "rezagos": ("Últimos errores conocidos de la estación (mismo horizonte y mismo horario de ayer)",
                "err_ult, err_media3, err_media6, err_ayer"),
    "sesgo_ewma": ("Sesgo habitual de la estación a esa hora, promedio exponencial con 4 memorias",
                   "sesgo_ewma_05, _10, _20, _30"),
    "vecinos": ("Errores recientes de las 5 estaciones más cercanas (< 100 km)", "vec_err_ult, vec_err_1h"),
    "revision": ("Cuánto cambió el pronóstico para esa hora entre emisiones", "rev_3h, rev_6h, rev_12h, rev_24h"),
    "rezagos_extra": ("Errores de 9 a 21 h antes y error del pronóstico a 1 h",
                      "err_hace9h … err_hace21h, err1h_ult, err1h_media2/4/8"),
    "dinamica": ("Tendencia, saltos y variabilidad reciente; error del punto de rocío",
                 "obs_tend3h, err1h_tend3h, salto_obs, err_rocio_ult, err_rocio_1h, err_std6, err_std24, "
                 "dispersion_emis, amplitud_pron"),
}
FAM_RIDGE = ["base", "pendiente", "rezagos", "sesgo_ewma", "revision"]
SIN_BARRA = {"displayModeBar": False}

st.set_page_config(page_title="Confianza del pronóstico · NWS Este", layout="wide")
st.markdown(f"""<style>
html, body, [class*="css"] {{ font-variant-numeric: tabular-nums; }}
.block-container {{ padding-top: 3rem; max-width: 1440px; }}
h1 {{ font-size: 24px !important; font-weight: 700 !important; margin: 0 !important; padding: 0 !important; }}
h3 {{ font-size: 18px !important; font-weight: 600 !important; margin-top: 8px !important; }}
.sub {{ color: {GRIS}; font-size: 14px; margin: 2px 0 0; }}
.hallazgo {{ color: {TINTA}; font-size: 14px; margin: 6px 0 0; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); background: {BLANCO}; border: 1px solid {BORDE};
        border-radius: 8px; }}
.kpi {{ padding: 14px 16px; border-right: 1px solid {BORDE}; }}
.kpi:last-child {{ border-right: none; }}
.kpi .l {{ color: {GRIS}; font-size: 14px; }}
.kpi .v {{ font-size: 30px; font-weight: 500; line-height: 1.25; }}
.kpi .c {{ color: {GRIS}; font-size: 12px; }}
.punto {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
.card-h {{ font-size: 16px; font-weight: 500; margin: 0; }}
.card-s {{ color: {GRIS}; font-size: 14px; margin: 2px 0 8px; }}
.nota {{ color: {GRIS}; font-size: 13px; margin: 6px 0 0; }}
.fila {{ display: flex; justify-content: space-between; padding: 11px 0; border-bottom: 1px solid {REGLA};
        font-size: 14px; }}
.fila .g {{ color: {GRIS}; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: {BLANCO}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{ font-size: 15px; padding: 8px 14px; }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- utilidades
def num(x, d=2):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def mae(r):
    return float(np.abs(r).mean()) if len(r) else np.nan


def estilo(fig, alto, hover="x unified"):
    fig.update_layout(height=alto, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor=BLANCO, plot_bgcolor=BLANCO,
                      font=dict(family="Geist, Inter, sans-serif", size=12, color=TINTA), hovermode=hover,
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=12, color=GRIS)))
    fig.update_xaxes(showgrid=False, linecolor=BORDE, tickfont=dict(color=GRIS, size=11))
    fig.update_yaxes(gridcolor=REGLA, zeroline=False, tickfont=dict(color=GRIS, size=11))
    return fig


def eje(fig, x=None, y=None):
    if x:
        fig.update_xaxes(title=dict(text=x, font=dict(color=GRIS, size=11)))
    if y:
        fig.update_yaxes(title=dict(text=y, font=dict(color=GRIS, size=11)))
    return fig


def tarjeta(titulo, sub=""):
    st.markdown(f'<p class="card-h">{titulo}</p>' + (f'<p class="card-s">{sub}</p>' if sub else ""),
                unsafe_allow_html=True)


def kpis(items):
    """items: lista de (etiqueta, valor, comentario, color del punto o None)."""
    html = "".join(
        f'<div class="kpi"><div class="l">' + (f'<span class="punto" style="background:{c}"></span>' if c else "")
        + f'{l}</div><div class="v">{v}</div><div class="c">{k}</div></div>' for l, v, k, c in items)
    st.markdown(f'<div class="kpis" style="grid-template-columns: repeat({len(items)}, 1fr)">{html}</div>',
                unsafe_allow_html=True)


def figura(ruta, pie):
    if ruta.exists():
        st.image(str(ruta), caption=pie, width="stretch")
    else:
        st.caption(f"(falta {ruta.relative_to(RAIZ)}: correr el pipeline para generarla)")


def tabla(df, **kw):
    st.dataframe(df, hide_index=True, width="stretch", **kw)


# ---------------------------------------------------------------- datos
@st.cache_resource
def cargar():
    D = pd.read_parquet(DATOS / "filas.parquet")
    D.insert(1, "target_time_utc", T0 + pd.to_timedelta(D.pop("horas_desde_t0"), unit="h"))
    B = pd.read_csv(DATOS / "bandas.csv", dtype={"tramo": "string", "condicion": "string"})
    D = D.join(B.set_index(["tramo", "condicion"]), on=["tramo", "condicion"])
    for c in B.columns[2:]:
        D[c] = D[c].astype("float32")
    E = pd.read_parquet(DATOS / "estaciones.parquet").set_index("station_id")
    D = D.join(E["estado"], on="station_id")
    for m in ("nws", "lineal", "lgbm"):                      # residuos: observado − pronóstico (corregido)
        D[f"e_{m}"] = (D["obs"] - D[m]).astype("float32")
    return D, E


@st.cache_data
def resultados():
    leer = lambda n: pd.read_csv(RES / n)
    return {n.stem: leer(n.name) for n in RES.glob("*.csv")}


@st.cache_data
def experimentos():
    return {n.stem: pd.read_csv(n, index_col=0 if n.stem == "redundancia_entre_familias" else None)
            for n in EXP.glob("*.csv")}


D, E = cargar()
R, X = resultados(), experimentos()
dias = D["target_time_utc"].dt.normalize()

# ---------------------------------------------------------------- encabezado
st.markdown("# Confianza del pronóstico de temperatura")
st.markdown(f'<p class="sub">NWS en {D.station_id.nunique()} estaciones del Este de EE.UU. · '
            f'{dias.nunique()} días de prueba entre el {dias.min():%d/%m} y el {dias.max():%d/%m/%Y}, que los '
            f'modelos no vieron al entrenar · corrección = observado − pronosticado, en °C</p>',
            unsafe_allow_html=True)
st.write("")

(t_hall, t_res, t_ridge, t_lgbm, t_vs, t_mon, t_var, t_val) = st.tabs(
    ["Hallazgos", "Resumen", "Ridge (Gram)", "LightGBM", "Ridge vs LightGBM", "Monitoreo", "Variables",
     "Datos y validación"])

# cifras comunes a varias pestañas (pliegues de reporte 3–5)
M = R["mae_horizonte"]
_F = M[(M.version == "final") & (M.segmento == "todas")]
mae_rep = lambda m, h="todos": float(_F[(_F.modelo == m) & (_F.horizonte == h)].MAE.iloc[0])


# ======================================================================== 0. HALLAZGOS (narrativa)
with t_hall:
    h0, hli, hlg = mae_rep("B0"), mae_rep("lineal"), mae_rep("lgbm")
    hb2 = mae_rep("B2")
    P_ = R["pruebas"].set_index(["segmento", "A", "B"]).loc[("todas", "lgbm", "lineal")]
    SR_ = R["sesgo_restante"].set_index("modelo")
    CB_ = R["cobertura_global"].set_index("modelo").cobertura
    LE_ = R["ridge_lectura"].iloc[0]
    st.markdown('<p class="hallazgo">Qué aprendimos al corregir el pronóstico de temperatura del NWS, en el orden en '
                'que lo contaríamos a quien tiene que decidir cuánto confiar en él.</p>', unsafe_allow_html=True)
    st.write("")
    kpis([("Error del NWS", f"{num(h0, 2)} °C", "MAE en días que los modelos no vieron", NWS),
          ("Con la mejor corrección", f"{num(hlg, 2)} °C", f"LightGBM · {num(100 * (h0 - hlg) / h0, 0)} % menos", AMBAR),
          ("A 1 h de antelación", f"−{num(100 * (1 - mae_rep('lgbm', '1') / mae_rep('B0', '1')), 0)} %",
           f"{num(mae_rep('B0', '1'))} → {num(mae_rep('lgbm', '1'))} °C", None),
          ("A 48 h de antelación", f"−{num(100 * (1 - mae_rep('lgbm', '48') / mae_rep('B0', '48')), 0)} %",
           f"{num(mae_rep('B0', '48'))} → {num(mae_rep('lgbm', '48'))} °C", None)])
    st.write("")
    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        st.markdown(f"""
#### 1. El NWS tiene un sesgo diario que se puede corregir
Su error medio cambia con la hora del día: a la mañana la temperatura real queda por encima del pronóstico y a la
tarde por debajo, con una amplitud de **{num(SR_.loc['B0', 'amplitud_hora'])} °C**. Las dos correcciones la bajan a
{num(SR_.loc['lineal', 'amplitud_hora'])} °C (Ridge) y {num(SR_.loc['lgbm', 'amplitud_hora'])} °C (LightGBM): el sesgo
que vimos en la Etapa A quedó capturado (gráfico de la derecha).

#### 2. Una tabla de promedios ya ayuda, pero el aprendizaje automático suma
Restar el error habitual de cada estación a cada hora (baseline B2) baja el error {num(100 * (h0 - hb2) / h0, 1)} %.
El Ridge lo baja {num(100 * (h0 - hli) / h0, 1)} % y LightGBM {num(100 * (h0 - hlg) / h0, 1)} %.

#### 3. LightGBM gana, por poco y de forma consistente
{num(-P_.dif_media_C, 3)} °C menos por día, {int(P_.dias_gana_A)} de {int(P_.dias)} días, p = {num(P_.p_valor, 4)}. Es
≈ 4 % del error del Ridge: el modelo de caja de cristal ya captura la mayor parte de la mejora.

#### 4. Por qué falla el pronóstico
Los dos modelos coinciden: el **ritmo del cambio** de temperatura pronosticado (frentes, salida y puesta del sol),
las **noches despejadas y calmas** ({num(LE_.enfriamiento_nocturno_C)} °C: enfría más de lo que el NWS anticipa), el
**viento**, el **sesgo propio de cada estación** y la **condición del tiempo** (lluvia).

#### 5. Cuánto confiar
La corrección rinde más cerca de la hora objetivo: el último error conocido todavía dice algo. La banda de confianza
del 80 % cubre en la práctica {num(100 * CB_['lineal'], 0)}–{num(100 * CB_['lgbm'], 0)} %: es algo optimista. Y la
mejora es en promedio: fila a fila, alrededor de 4 de cada 10 pronósticos corregidos quedan algo peor que el original.

#### 6. Límites
15 días de prueba de otoño en una región; el modelo no vio invierno ni verano. Entre el 19 y el 26/10 el NWS falló al
publicar observaciones (del 23 al 26, 272 estaciones sin ningún dato): esos 7 días se excluyeron con una regla
automática. La condición del tiempo
de los filtros es la **pronosticada**, no la observada.

**Recomendación:** usar LightGBM para corregir (menor error) y el Ridge para explicar y auditar la corrección; mostrar
siempre la banda y el horizonte, porque a 48 h la mejora es menos de la mitad que a 1 h.
""")
    with c2, st.container(border=True):
        tarjeta("Error medio por hora del día", "Observado − pronóstico, en °C (todas las filas de prueba). Por "
                "encima de 0, la temperatura real fue más alta que la pronosticada.")
        g = D.groupby("hora_solar")[["e_nws", "e_lineal", "e_lgbm"]].mean()
        fig = go.Figure()
        for col, k in (("e_nws", "nws"), ("e_lineal", "lineal"), ("e_lgbm", "lgbm")):
            fig.add_trace(go.Scatter(x=g.index, y=g[col], name=NOMBRE[k], mode="lines+markers",
                                     line=dict(color=COLOR[k], width=2),
                                     marker=dict(size=7, line=dict(width=2, color=BLANCO))))
        fig.add_hline(y=0, line=dict(color=BORDE, width=1))
        fig.update_xaxes(dtick=3)
        st.plotly_chart(eje(estilo(fig, 320), "hora solar local", "error medio (°C)"), width="stretch",
                        config=SIN_BARRA)
        tarjeta("Error según la antelación", "MAE en °C, pliegues de reporte.")
        G = M[(M.version == "final") & (M.segmento == "todas") & (M.horizonte != "todos")].copy()
        G["h"] = G.horizonte.astype(int)
        fig = go.Figure()
        for k in ("B0", "lineal", "lgbm"):
            g2 = G[G.modelo == k].sort_values("h")
            fig.add_trace(go.Scatter(x=g2.h, y=g2.MAE, name=NOMBRE[k], mode="lines+markers",
                                     line=dict(color=COLOR[k], width=2),
                                     marker=dict(size=7, line=dict(width=2, color=BLANCO))))
        st.plotly_chart(eje(estilo(fig, 260), "horizonte (h)", "MAE (°C)"), width="stretch", config=SIN_BARRA)



# ======================================================================== 1. RESUMEN
with t_res:
    M = R["mae_horizonte"]
    F0 = M[(M.version == "final") & (M.horizonte == "todos")]
    v = lambda seg, m: float(F0[(F0.segmento == seg) & (F0.modelo == m)].MAE.iloc[0])
    b0, b2, li, lg = (v("todas", m) for m in ("B0", "B2", "lineal", "lgbm"))
    P = R["pruebas"].set_index(["segmento", "A", "B"])
    p_ll = P.loc[("todas", "lgbm", "lineal")]
    st.markdown(f'<p class="hallazgo">Las dos correcciones le ganan al NWS y a la tabla de promedios (B2). '
                f'LightGBM gana por poco pero de forma consistente: {num(-p_ll.dif_media_C, 3)} °C menos por día, '
                f'{int(p_ll.dias_gana_A)} de {int(p_ll.dias)} días, p = {num(p_ll.p_valor, 4)}. El Ridge ya captura '
                f'{num(100 * (b0 - li) / (b0 - lg), 0)} % de la mejora que logra LightGBM sobre el NWS.</p>',
                unsafe_allow_html=True)
    st.write("")
    kpis([("NWS crudo (B0)", f"{num(b0, 3)} °C", "MAE, pliegues de reporte 3–5", NWS),
          ("Estación × hora (B2)", f"{num(b2, 3)} °C", f"{num(100 * (b0 - b2) / b0, 1)} % menos que el NWS", B2C),
          ("Ridge (Gram)", f"{num(li, 3)} °C", f"{num(100 * (b0 - li) / b0, 1)} % menos · 5 familias, α = 10⁵", AZUL),
          ("LightGBM", f"{num(lg, 3)} °C", f"{num(100 * (b0 - lg) / b0, 1)} % menos · 8 familias, 150 árboles",
           AMBAR)])
    st.write("")

    c1, c2 = st.columns(2, gap="small")
    with c1, st.container(border=True):
        tarjeta("Error según el horizonte", "MAE en °C en los pliegues de reporte. Las dos correcciones ganan más "
                "a horizontes cortos, donde el error reciente todavía dice algo.")
        seg = st.radio("Días de prueba", ["todas", "completa", "incompleta"], horizontal=True, key="res_seg",
                       format_func={"todas": "Todos", "completa": "Historia completa",
                                    "incompleta": "Historia incompleta"}.get)
        G = M[(M.version == "final") & (M.segmento == seg) & (M.horizonte != "todos")].copy()
        G["h"] = G.horizonte.astype(int)
        fig = go.Figure()
        for m in ("B0", "B2", "lineal", "lgbm"):
            g = G[G.modelo == m].sort_values("h")
            fig.add_trace(go.Scatter(x=g.h, y=g.MAE, name=NOMBRE[m], mode="lines+markers",
                                     line=dict(color=COLOR[m], width=2, dash="dot" if m == "B2" else "solid"),
                                     marker=dict(size=8, line=dict(width=2, color=BLANCO))))
        st.plotly_chart(eje(estilo(fig, 300), "horizonte (h)", "MAE (°C)"), width="stretch", config=SIN_BARRA)
    with c2, st.container(border=True):
        tarjeta("Error por día de prueba", "Cada día es una unidad de la prueba de significancia: dentro de un "
                "día los errores de estaciones y horizontes están correlacionados.")
        Dd = R["mae_por_dia"]
        etiqueta = pd.to_datetime(Dd.dia).dt.strftime("%d/%m") + np.where(Dd.historia == "incompleta", "*", "")
        fig = go.Figure()
        for m in ("B0", "B2", "lineal", "lgbm"):
            fig.add_trace(go.Scatter(x=etiqueta, y=Dd[m], name=NOMBRE[m], mode="lines+markers",
                                     line=dict(color=COLOR[m], width=2, dash="dot" if m == "B2" else "solid"),
                                     marker=dict(size=8, line=dict(width=2, color=BLANCO))))
        fig.update_xaxes(type="category")
        st.plotly_chart(eje(estilo(fig, 300), "día de prueba (* = historia incompleta)", "MAE (°C)"),
                        width="stretch", config=SIN_BARRA)

    with st.container(border=True):
        tarjeta("¿Las diferencias superan el ruido?", "t pareado sobre la diferencia de MAE diario (Diebold–Mariano "
                "con pérdida absoluta) e intervalo bootstrap remuestreando días. Negativo = A es mejor.")
        T = R["pruebas"].copy()
        T["comparación"] = T.A.map(NOMBRE) + " − " + T.B.map(NOMBRE)
        T["días que gana A"] = T.dias_gana_A.astype(str) + " de " + T.dias.astype(str)
        T = T.rename(columns={"segmento": "días de prueba", "dif_media_C": "dif. media (°C)",
                              "IC95_inf": "IC95 inf", "IC95_sup": "IC95 sup", "p_valor": "p"})
        tabla(T[["días de prueba", "comparación", "dif. media (°C)", "IC95 inf", "IC95 sup", "t", "p",
                 "días que gana A"]],
              column_config={"dif. media (°C)": st.column_config.NumberColumn(format="%.3f"),
                             "IC95 inf": st.column_config.NumberColumn(format="%.3f"),
                             "IC95 sup": st.column_config.NumberColumn(format="%.3f"),
                             "t": st.column_config.NumberColumn(format="%.2f"),
                             "p": st.column_config.NumberColumn(format="%.4f")})
        st.markdown('<p class="nota">Lectura: LightGBM − Ridge es negativo en los tres segmentos y su intervalo no '
                    'toca el cero. La diferencia es real pero chica (≈ 4 % del MAE del Ridge). Con 15 días de '
                    'otoño en una región, la conclusión vale para ese período.</p>', unsafe_allow_html=True)


# ======================================================================== 2. RIDGE (GRAM)
with t_ridge:
    A_ = R["ridge_alpha"]
    fin = A_[(A_.version.str.startswith("Ridge final")) & A_.elegido].iloc[0]
    lect = R["ridge_lectura"].iloc[0]
    st.markdown('<p class="hallazgo">Regresión lineal con penalización L2 (Ridge) y curvas (splines) en las '
                'variables con efecto no lineal. Cada variable aporta un coeficiente o una curva legible en °C. '
                '«Gram»: todas las variantes y penalizaciones se resuelven desde una sola matriz XᵀX por pliegue '
                '(<code>src/ridge_gram.py</code>), con el mismo resultado que reentrenar scikit-learn.</p>',
                unsafe_allow_html=True)
    st.write("")
    kpis([("MAE de reporte", f"{num(li, 3)} °C", "pliegues 3–5, todos los días", AZUL),
          ("Penalización elegida", "α = 10⁵", f"MAE de elección {num(fin.MAE_eleccion, 4)} °C (pliegues 1–2)", None),
          ("Familias", "5 de 8", "salen vecinos, rezagos extra y dinámica", None),
          ("Enfriamiento nocturno", f"{num(lect.enfriamiento_nocturno_C)} °C", "noche × despejado × calma", None)])
    st.write("")

    st.markdown("### Versiones")
    tabla(pd.DataFrame([
        ["v1 · Elastic Net (rama main)", "todas las variables", "L1 + L2", "1,286", "—",
         "Descartado: la proporción L1/L2 no cambiaba el error; coeficientes de signo opuesto en humedad y rocío"],
        ["v2 · Ridge (rama main)", "todas las variables", "L2", "1,290", "—",
         "Mismo error que el Elastic Net y más legible: se adopta Ridge"],
        ["v3 · Ridge con las 8 familias", "8 familias", "L2, α = 10⁵",
         num(A_[(A_.version == "Ridge con las 8 familias") & A_.elegido].MAE_eleccion.iloc[0], 4),
         num(float(M[(M.version == "ridge_8_familias") & (M.segmento == "todas")
                     & (M.horizonte == "todos")].MAE.iloc[0]), 3),
         "Punto de partida de la ablación"],
        ["v4 · Ridge final (producto)", ", ".join(FAM_RIDGE), "L2, α = 10⁵", num(fin.MAE_eleccion, 4), num(li, 3),
         "Regla de recorte: mismo error que v3 con 3 familias menos"],
    ], columns=["versión", "variables", "regularización", "MAE elección (°C)", "MAE reporte (°C)", "decisión"]))
    st.markdown('<p class="nota">v1 y v2 se midieron en la rama main con otros pliegues: sus números no son '
                'comparables con los de v3 y v4, solo explican la elección del tipo de modelo.</p>',
                unsafe_allow_html=True)

    st.markdown("### Búsqueda de la penalización α")
    c1, c2 = st.columns(2, gap="small")
    with c1, st.container(border=True):
        tarjeta("MAE de elección según α", "Promedio de los pliegues 1–2. Con ~10⁶ filas estandarizadas α chico casi "
                "no penaliza; por eso la grilla va de 1 a 10⁷. Las dos versiones eligen 10⁵, en el interior.")
        fig = go.Figure()
        for ver, col, dash in (("Ridge final (5 familias)", AZUL, "solid"), ("Ridge con las 8 familias", GRIS, "dot")):
            g = A_[A_.version == ver]
            fig.add_trace(go.Scatter(x=g.alpha, y=g.MAE_eleccion, name=ver, mode="lines+markers",
                                     line=dict(color=col, width=2, dash=dash),
                                     marker=dict(size=8, line=dict(width=2, color=BLANCO))))
        fig.add_vline(x=1e5, line=dict(color=AZUL, width=1, dash="dash"))
        fig.update_xaxes(type="log", tickvals=[1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
                         ticktext=["1", "10²", "10³", "10⁴", "10⁵", "10⁶", "10⁷"])
        st.plotly_chart(eje(estilo(fig, 300), "α (penalización L2)", "MAE (°C)"), width="stretch",
                        config=SIN_BARRA)
    with c2, st.container(border=True):
        tarjeta("α para cada variante de la ablación", "Cada variante saca una familia y elige su propio α "
                "(promedio de los pliegues 1–2). Todas se resolvieron desde la misma XᵀX.")
        RV = X["ridge_variantes_por_alpha"].groupby(["variante", "alpha"], as_index=False).MAE.mean()
        sel = st.multiselect("Variantes", sorted(RV.variante.unique()), default=["todas", "sin base", "sin sesgo_ewma",
                                                                                  "sin vecinos"], key="rv")
        fig = go.Figure()
        for var in sel:
            g = RV[RV.variante == var]
            destacada = var == "todas"
            fig.add_trace(go.Scatter(x=g.alpha, y=g.MAE, name=var, mode="lines+markers",
                                     line=dict(color=AZUL if destacada else GRIS, width=2.4 if destacada else 1.2,
                                               dash="solid" if destacada else "dot"),
                                     marker=dict(size=8 if destacada else 6)))
        fig.update_xaxes(type="log", tickvals=[1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
                         ticktext=["1", "10²", "10³", "10⁴", "10⁵", "10⁶", "10⁷"])
        st.plotly_chart(eje(estilo(fig, 260), "α", "MAE (°C)"), width="stretch", config=SIN_BARRA)

    st.markdown("### Variables: qué familias entran (regla de recorte)")
    c1, c2 = st.columns([1.15, 1], gap="small")
    with c1, st.container(border=True):
        tarjeta("Cuánto empeora el Ridge sin cada familia", "Δ MAE en °C (pliegues de elección). Una familia sale si "
                "sin ella el promedio no empeora más de 0,005 °C y ningún pliegue más de 0,010 °C. La regla se fijó "
                "antes de ver los resultados.")
        Ab = X["ablacion"].sort_values("delta_promedio")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=Ab.familia, x=Ab.delta_promedio, orientation="h", name="Δ promedio",
                             marker=dict(color=[GRIS if s else AZUL for s in Ab.se_saca], cornerradius=4),
                             hovertemplate="%{y}: %{x:.4f} °C<extra>Δ promedio</extra>"))
        fig.add_trace(go.Scatter(y=Ab.familia, x=Ab.delta_max, mode="markers", name="Δ peor pliegue",
                                 marker=dict(symbol="line-ns-open", size=14, color=TINTA, line=dict(width=2)),
                                 hovertemplate="%{y}: %{x:.4f} °C<extra>peor pliegue</extra>"))
        fig.add_vline(x=0.005, line=dict(color=GRIS, width=1, dash="dash"), annotation_text="0,005", annotation_position="bottom right",
                      annotation_font=dict(color=GRIS, size=11))
        fig.add_vline(x=0.010, line=dict(color=GRIS, width=1, dash="dot"), annotation_text="0,010", annotation_position="bottom right",
                      annotation_font=dict(color=GRIS, size=11))
        st.plotly_chart(eje(estilo(fig, 320, "y unified"), "Δ MAE sin la familia (°C)"), width="stretch",
                        config=SIN_BARRA)
        st.markdown('<p class="nota">Azul = queda en el Ridge; gris = sale. Revisión queda porque en el pliegue 2 '
                    'empeora 0,014 °C. Sacar las tres juntas tampoco empeora (1,247 contra 1,246).</p>',
                    unsafe_allow_html=True)
    with c2, st.container(border=True):
        tarjeta("Familias del modelo")
        tabla(pd.DataFrame([{"familia": f, "en el Ridge": "sí" if f in FAM_RIDGE else "no",
                             "qué mide": FAMILIAS[f][0], "variables": FAMILIAS[f][1]} for f in FAMILIAS]), height=330)
        Ap = X["aporte_por_familia"].groupby("familia")[["aporte_desvio_C", "coef_que_cambian_de_signo_pct"]].mean()
        st.markdown(f'<p class="nota">Rezagos extra mueve la predicción (aporte {num(Ap.loc["rezagos_extra", "aporte_desvio_C"])} °C) '
                    f'pero repite a los rezagos (redundancia 0,87): por eso sacarla no cambia el error.</p>',
                    unsafe_allow_html=True)

    st.markdown("### Regularización y estabilidad")
    c1, c2 = st.columns(2, gap="small")
    with c1, st.container(border=True):
        tarjeta("Cómo se regulariza", "Lo que controla la complejidad del Ridge, además de α.")
        tabla(pd.DataFrame([
            ["Penalización L2 (α)", "10⁵ en la grilla 1–10⁷", "Achica los coeficientes; elegida en los pliegues 1–2 y congelada"],
            ["Selección de familias", "regla de recorte (0,005 / 0,010 °C)", "Saca lo que no mejora: menos columnas, más legible"],
            ["Splines", "hora solar (periódica), pendiente, temp_anom, dep_rocio, viento", "Curvas suaves en lugar de una recta"],
            ["Interacciones fijadas por física", "noche × despejado × calma; sesgo × noche; err_ult × horizonte",
             "Umbrales elegidos por criterio físico, no ajustados con datos"],
            ["Faltantes", "0 + indicador (lineales); mediana (curvas)", "Ajustados solo con el entrenamiento"],
            ["Submuestra", "10⁶ filas por pliegue", "La matriz es ancha; 8 GB de RAM"],
        ], columns=["mecanismo", "valor", "efecto"]))
    with c2, st.container(border=True):
        tarjeta("Aporte de cada familia a la predicción", "Desvío del aporte en °C (barra) y % de coeficientes que "
                "cambian de signo entre los pliegues 1 y 2 (en el texto). Un signo inestable es una lectura poco "
                "confiable.")
        Ap = Ap.sort_values("aporte_desvio_C")
        fig = go.Figure(go.Bar(y=Ap.index, x=Ap.aporte_desvio_C, orientation="h",
                               marker=dict(color=[AZUL if f in FAM_RIDGE else GRIS for f in Ap.index], cornerradius=4),
                               text=[f"{num(s, 0)} % cambian de signo" for s in Ap.coef_que_cambian_de_signo_pct],
                               textposition="outside", textfont=dict(color=GRIS, size=11),
                               hovertemplate="%{y}: %{x:.3f} °C<extra></extra>"))
        fig.update_xaxes(range=[0, Ap.aporte_desvio_C.max() * 1.6])
        st.plotly_chart(eje(estilo(fig, 300, "closest"), "desvío del aporte (°C)"), width="stretch",
                        config=SIN_BARRA)

    st.markdown("### Interpretación (Etapa B)")
    c1, c2 = st.columns([1.4, 1], gap="small")
    with c1:
        figura(FIG / "interp_curvas_ridge.png", "Curvas del Ridge: aporte de cada variable a la corrección, en °C "
               "(último pliegue de reporte).")
    with c2, st.container(border=True):
        tarjeta("Lo que dice el modelo")
        st.markdown(f"""
- **Enfriamiento nocturno:** {num(lect.enfriamiento_nocturno_C)} °C en noches despejadas y calmas (inversión por
  radiación que el NWS subestima).
- **Pendiente del pronóstico:** casi lineal, de −0,8 a +0,9 °C. Si el pronóstico sube rápido, la observación queda
  por encima.
- **Hora solar:** −0,4 °C a las 15 h y +0,2 °C de madrugada, además del sesgo exponencial.
- **Peso del último error (`err_ult`):** {num(lect.peso_err_ult_1h)} °C por °C a 1 h, {num(lect.peso_err_ult_6h)} a
  6 h, {num(lect.peso_err_ult_24h)} a 24 h y {num(lect.peso_err_ult_48h)} a 48 h: la persistencia se apaga con la
  antelación.
- **Estabilidad:** los coeficientes de dos pliegues de reporte correlacionan 0,956.
""")
    with st.expander("Ablación y aporte por familia (figuras de la notebook de experimentos)"):
        c1, c2 = st.columns(2)
        with c1:
            figura(FIG_EXP / "ablacion.png", "Ablación: Δ MAE sin cada familia.")
        with c2:
            figura(FIG_EXP / "aporte_por_familia.png", "Aporte de cada familia a la predicción del Ridge.")
        Dv = X["detalle_variantes"]
        Dv = Dv[(Dv.horizonte == "todos") & (Dv.segmento == "completa")].pivot_table(
            index="pliegue", columns="modelo", values="MAE").reset_index()
        st.markdown('<p class="card-s">MAE (°C) en los pliegues de elección, historia completa:</p>',
                    unsafe_allow_html=True)
        tabla(Dv.round(4))


# ======================================================================== 3. LIGHTGBM
with t_lgbm:
    TOP = R["lgbm_top5"]
    IMP = R["lgbm_importancia_familia"]
    st.markdown('<p class="hallazgo">Gradient boosting de árboles con las 8 familias (51 variables). No necesita '
                'imputar faltantes ni declarar curvas o interacciones: las aprende. A cambio, se explica con SHAP '
                'en lugar de coeficientes.</p>', unsafe_allow_html=True)
    st.write("")
    kpis([("MAE de reporte", f"{num(lg, 3)} °C", "pliegues 3–5, todos los días", AMBAR),
          ("Hiperparámetros", "127 hojas · L1", f"150 árboles · MAE de elección {num(TOP.MAE_eleccion.iloc[0], 4)} °C",
           None),
          ("Variables", "51", "las 8 familias; condición como categórica", None),
          ("Ganancia de la base", f"{num(IMP.set_index('familia').ganancia_pct['base'], 1)} %",
           "sesgo EWMA y pendiente le siguen", None)])
    st.write("")

    st.markdown("### Versiones")
    tabla(pd.DataFrame([
        ["v1 · familias 1–4 (rama main)", "base, pendiente, rezagos, sesgo EWMA", "grilla 250–2000 árboles",
         "1,325", "El óptimo quedó en el borde de la grilla: con ~20 días de entrenamiento se sobreajusta rápido"],
        ["v2 · grilla fina (rama main)", "base, pendiente, rezagos, sesgo EWMA", "50–500 árboles, de a 50",
         "—", "Cortes finos para encontrar el óptimo en el interior"],
        ["v3 · experimento --todas (16/11)", "las 8 familias", "50–500 árboles", "1,313",
         "−0,9 %, mejora 6 de 10 días, p = 0,12: la ganancia está a 1–6 h"],
        ["v4 · LightGBM final (producto)", "las 8 familias (51 variables)", "127 hojas, L1, 150 árboles",
         num(lg, 3), "La pregunta del producto: ¿una caja negra con toda la información gana con claridad?"],
    ], columns=["versión", "variables", "búsqueda", "MAE reporte (°C)", "decisión"]))
    st.markdown('<p class="nota">v1–v3 se midieron en la rama main con los pliegues 3–4 de entonces: no son '
                'comparables con v4 (pliegues anclados 3–5), solo muestran cómo se llegó a la versión final.</p>',
                unsafe_allow_html=True)

    st.markdown("### Búsqueda de hiperparámetros")
    c1, c2 = st.columns([1, 1.1], gap="small")
    with c1, st.container(border=True):
        tarjeta("Grilla", "num_leaves ∈ {31, 127} × pérdida ∈ {L2, L1} × árboles de 50 a 500 (cortes cada 50). "
                "Criterio: MAE con historia completa en los pliegues 1–2; queda congelado.")
        Tm = R["lgbm_tiempos"].pivot_table(index="combinacion", columns="pliegue", values="segundos").reset_index()
        Tm.columns = ["combinación", "seg. pliegue 1", "seg. pliegue 2"]
        Tm["hojas"] = Tm["combinación"].str.extract(r"(\d+)").astype(int)
        Tm["pérdida"] = np.where(Tm["combinación"].str.endswith("_l1"), "L1", "L2")
        Tm["en el top 5"] = Tm["combinación"].map(TOP.groupby("combinacion").size()).fillna(0).astype(int)
        tabla(Tm[["hojas", "pérdida", "en el top 5", "seg. pliegue 1", "seg. pliegue 2"]])
        st.markdown('<p class="nota">Las 5 mejores combinaciones usan 127 hojas. L1 gana porque el criterio es el MAE; '
                    'entre 150 y 250 árboles la diferencia es de 0,001 °C: la elección es estable.</p>',
                    unsafe_allow_html=True)
    with c2, st.container(border=True):
        tarjeta("Las 5 mejores combinaciones", "MAE de elección en °C (pliegues 1–2). El eje empieza cerca del valor "
                "para que se vea la diferencia, que es de milésimas.")
        TOP = TOP.assign(etq=TOP.num_leaves.astype(str) + " hojas · " + TOP.objetivo.str[:2] + " · "
                         + TOP.arboles.astype(str) + " árboles")
        fig = go.Figure(go.Bar(y=TOP.etq[::-1], x=TOP.MAE_eleccion[::-1], orientation="h",
                               marker=dict(color=[AMBAR if p == 1 else GRIS for p in TOP.puesto[::-1]],
                                           cornerradius=4),
                               text=[num(x, 4) for x in TOP.MAE_eleccion[::-1]], textposition="outside",
                               textfont=dict(color=GRIS, size=11), hovertemplate="%{y}: %{x:.4f} °C<extra></extra>"))
        fig.update_xaxes(range=[1.2115, 1.2142])
        st.plotly_chart(eje(estilo(fig, 280, "closest"), "MAE de elección (°C)"), width="stretch",
                        config=SIN_BARRA)

    st.markdown("### Regularización")
    c1, c2 = st.columns(2, gap="small")
    with c1, st.container(border=True):
        tarjeta("Cómo se controla el sobreajuste", "Los fijos usan valores usuales y no se ajustaron.")
        tabla(pd.DataFrame([
            ["Cantidad de árboles", "150 (de 50–500)", "elegido", "Corte temprano: más árboles sobreajustan"],
            ["num_leaves", "127 (de 31 / 127)", "elegido", "Complejidad de cada árbol"],
            ["Pérdida", "L1 (de L2 / L1)", "elegido", "Robusta a errores grandes; alineada con el MAE"],
            ["learning_rate", "0,05", "fijo", "Pasos chicos: cada árbol corrige poco"],
            ["min_child_samples", "500", "fijo", "Las filas se repiten por horizonte: una hoja chica es una sola situación"],
            ["feature_fraction", "0,8", "fijo", "Cada árbol ve el 80 % de las variables"],
            ["bagging_fraction", "0,8", "fijo", "Cada árbol ve el 80 % de las filas"],
            ["lambda_l2", "1", "fijo", "Penalización L2 sobre los valores de las hojas"],
            ["Submuestra", "2·10⁶ filas por pliegue", "fijo", "Memoria y tiempo"],
        ], columns=["parámetro", "valor", "cómo", "efecto"]))
    with c2, st.container(border=True):
        tarjeta("Importancia por familia", "Fracción de la ganancia de los árboles, media de los pliegues de reporte.")
        I = IMP.sort_values("ganancia_pct")
        fig = go.Figure(go.Bar(y=I.familia, x=I.ganancia_pct, orientation="h",
                               marker=dict(color=AMBAR, cornerradius=4),
                               text=[f"{num(x, 1)} %" for x in I.ganancia_pct], textposition="outside",
                               textfont=dict(color=GRIS, size=11), hovertemplate="%{y}: %{x:.1f} %<extra></extra>"))
        fig.update_xaxes(range=[0, I.ganancia_pct.max() * 1.18])
        st.plotly_chart(eje(estilo(fig, 330, "closest"), "% de la ganancia"), width="stretch", config=SIN_BARRA)

    st.markdown("### Interpretación con SHAP (Etapa B)")
    c1, c2 = st.columns([1, 1.3], gap="small")
    with c1, st.container(border=True):
        tarjeta("Las 10 variables con más peso", "|contribución| SHAP media en °C (TreeSHAP exacto, 100 000 filas de "
                "prueba del último pliegue).")
        S = R["shap_top10"].sort_values("shap_abs_medio_C")
        fig = go.Figure(go.Bar(y=S.variable, x=S.shap_abs_medio_C, orientation="h",
                               marker=dict(color=AMBAR, cornerradius=4),
                               hovertemplate="%{y}: %{x:.3f} °C<extra></extra>"))
        st.plotly_chart(eje(estilo(fig, 340, "closest"), "|SHAP| medio (°C)"), width="stretch", config=SIN_BARRA)
        st.markdown(f'<p class="nota">Latitud y longitud no funcionan como identificador de estación: su aporte por '
                    f'estación correlaciona {num(lect.r_latlon_sesgo)} con el sesgo histórico; los sesgos '
                    f'exponenciales, {num(lect.r_ewma_sesgo)}.</p>', unsafe_allow_html=True)
    with c2:
        figura(FIG / "interp_shap_lgbm.png", "SHAP de LightGBM.")
        figura(FIG / "interp_latlon.png", "Aporte de latitud y longitud por estación contra su sesgo histórico.")


# ======================================================================== 4. RIDGE vs LIGHTGBM
with t_vs:
    st.markdown('<p class="hallazgo">Las dos correcciones sobre las mismas filas de prueba. Usá los filtros para '
                'ver dónde la diferencia crece o desaparece.</p>', unsafe_allow_html=True)
    f = st.columns([1.2, 1.2, 1.6, 1.6])
    h_rng = f[0].slider("Horizonte (h)", 1, 48, (1, 48), key="vs_h")
    hist = f[1].selectbox("Días de prueba", ["todos", "completa", "incompleta"], key="vs_hist",
                          format_func={"todos": "Todos", "completa": "Historia completa",
                                       "incompleta": "Historia incompleta"}.get)
    conds = f[2].multiselect("Condición del tiempo", list(CONDICIONES), format_func=CONDICIONES.get,
                             placeholder="Todas", key="vs_cond")
    estados = f[3].multiselect("Estados", sorted(D["estado"].dropna().unique()), placeholder="Todos", key="vs_est")
    m = D["horizonte"].between(*h_rng).to_numpy(copy=True)
    if hist != "todos":
        m &= (D["historia"] == hist).to_numpy()
    if conds:
        m &= D["condicion"].isin(conds).to_numpy()
    if estados:
        m &= D["estado"].isin(estados).to_numpy()
    F = D[m]
    if F.empty:
        st.info("Ninguna fila cumple estos filtros.")
    else:
        def metricas(e, base):
            a, ab = np.abs(e.to_numpy(np.float64)), np.abs(base.to_numpy(np.float64))
            return {"MAE": a.mean(), "RMSE": np.sqrt((a ** 2).mean()), "sesgo": float(e.mean()),
                    "mediana |e|": np.median(a), "P90 |e|": np.quantile(a, .9), "% |e| > 3 °C": 100 * (a > 3).mean(),
                    "% filas que empeora": 100 * (a > ab + 1e-6).mean()}
        T = pd.DataFrame({NOMBRE[k]: metricas(F[f"e_{k}"], F["e_nws"]) for k in ("nws", "lineal", "lgbm")}).T
        T.loc[NOMBRE["nws"], "% filas que empeora"] = np.nan
        T["mejora MAE vs NWS %"] = 100 * (1 - T["MAE"] / T.loc[NOMBRE["nws"], "MAE"])
        e_li, e_lg = T.loc["Ridge", "MAE"], T.loc["LightGBM", "MAE"]
        kpis([("Ridge", f"{num(e_li, 3)} °C", f"MAE · {num(T.loc['Ridge', 'mejora MAE vs NWS %'], 1)} % menos que el NWS",
               AZUL),
              ("LightGBM", f"{num(e_lg, 3)} °C",
               f"MAE · {num(T.loc['LightGBM', 'mejora MAE vs NWS %'], 1)} % menos que el NWS", AMBAR),
              ("Ventaja de LightGBM", f"{num(e_li - e_lg, 3)} °C", f"{num(100 * (e_li - e_lg) / e_li, 1)} % del MAE del Ridge",
               None),
              ("Filas comparadas", num(len(F), 0), "observaciones reales de prueba", None)])
        st.write("")
        with st.container(border=True):
            tarjeta("Métricas en la selección", "En °C salvo los porcentajes. «% filas que empeora»: fila a fila, la "
                    "corrección queda más lejos de lo observado que el NWS crudo.")
            Tf = T.copy().astype(object)
            for c in T.columns:
                d = 3 if c in ("MAE", "RMSE", "sesgo", "mediana |e|", "P90 |e|") else 1
                Tf[c] = [("—" if pd.isna(x) else num(x, d)) for x in T[c]]
            tabla(Tf.reset_index(names="modelo"))

        Fa = F.assign(a=F.e_nws.abs(), b=F.e_lineal.abs(), c=F.e_lgbm.abs())
        c1, c2 = st.columns(2, gap="small")
        with c1, st.container(border=True):
            tarjeta("Según la condición del tiempo", "MAE en °C. Grupos con menos de 5.000 filas no se muestran.")
            g = Fa.groupby("condicion", observed=True)[["a", "b", "c"]].mean()
            g = g[Fa["condicion"].value_counts().reindex(g.index) >= 5000].sort_values("c")
            fig = go.Figure()
            for col, k in (("a", "nws"), ("b", "lineal"), ("c", "lgbm")):
                fig.add_trace(go.Bar(x=[CONDICIONES.get(i, i) for i in g.index], y=g[col], name=NOMBRE[k],
                                     marker=dict(color=COLOR[k], cornerradius=4)))
            fig.update_layout(bargap=0.25, bargroupgap=0.08)
            st.plotly_chart(eje(estilo(fig, 300), None, "MAE (°C)"), width="stretch", config=SIN_BARRA)
        with c2, st.container(border=True):
            tarjeta("Según la hora del día", "MAE en °C por hora solar local. El NWS tiene un ciclo diario que los "
                    "modelos casi eliminan.")
            g = Fa.groupby("hora_solar")[["a", "b", "c"]].mean()
            fig = go.Figure()
            for col, k in (("a", "nws"), ("b", "lineal"), ("c", "lgbm")):
                fig.add_trace(go.Scatter(x=g.index, y=g[col], name=NOMBRE[k], line=dict(color=COLOR[k], width=2)))
            fig.update_xaxes(dtick=3)
            st.plotly_chart(eje(estilo(fig, 300), "hora solar", "MAE (°C)"), width="stretch", config=SIN_BARRA)

        c1, c2 = st.columns([1.1, 1], gap="small")
        with c1, st.container(border=True):
            tarjeta("¿Qué modelo gana en cada estación?", "MAE del Ridge − MAE de LightGBM, en °C. Ámbar: gana "
                    "LightGBM; azul: gana el Ridge; gris: empatan.")
            pe = Fa.groupby("station_id", observed=True)[["b", "c"]].mean().join(E[["estado", "lat", "lon"]])
            pe["n"] = Fa.groupby("station_id", observed=True).size()
            pe = pe[pe.n >= 50]
            pe["dif"] = pe.b - pe.c
            lim = float(np.nanpercentile(np.abs(pe.dif), 95)) or 0.1
            fig = go.Figure(go.Scattergeo(
                lon=pe.lon, lat=pe.lat, text=pe.index + " · " + pe.estado.astype(str), mode="markers",
                marker=dict(size=7, color=pe.dif, cmin=-lim, cmax=lim, line=dict(width=0.5, color=BLANCO),
                            colorscale=[[0, AZUL], [0.5, "#d4d4d4"], [1, AMBAR]],
                            colorbar=dict(title=dict(text="°C", font=dict(size=11, color=GRIS)), thickness=10,
                                          tickfont=dict(size=10, color=GRIS))),
                hovertemplate="%{text}<br>Ridge − LightGBM: %{marker.color:.3f} °C<extra></extra>"))
            fig.update_geos(scope="north america", projection_type="mercator", showland=True, landcolor="#f7f7f7",
                            showsubunits=True, subunitcolor=BORDE, countrycolor=BORDE, showlakes=False,
                            lonaxis_range=[-92, -66], lataxis_range=[24, 48], bgcolor=BLANCO)
            st.plotly_chart(estilo(fig, 380, "closest"), width="stretch", config=SIN_BARRA)
            st.markdown(f'<p class="nota">LightGBM gana en {(pe.dif > 0).sum()} de {len(pe)} estaciones '
                        f'(con al menos 50 filas en la selección).</p>', unsafe_allow_html=True)
        with c2, st.container(border=True):
            tarjeta("¿Aprenden lo mismo?", "Corrección del Ridge contra la de LightGBM (muestra de 20.000 filas). "
                    "Si estuvieran sobre la diagonal, LightGBM sería un Ridge más flexible y nada más.")
            mu = F.sample(min(20_000, len(F)), random_state=0)
            cl, cg = mu.lineal - mu.nws, mu.lgbm - mu.nws
            r = float(np.corrcoef(F.lineal - F.nws, F.lgbm - F.nws)[0, 1])
            lim2 = float(np.nanpercentile(np.abs(np.r_[cl, cg]), 99.5))
            fig = go.Figure([go.Scattergl(x=cl, y=cg, mode="markers", name="filas",
                                          marker=dict(size=4, color=TINTA, opacity=0.15),
                                          hovertemplate="Ridge %{x:.2f} · LightGBM %{y:.2f} °C<extra></extra>"),
                             go.Scatter(x=[-lim2, lim2], y=[-lim2, lim2], mode="lines", name="diagonal",
                                        line=dict(color=GRIS, width=1, dash="dash"), hoverinfo="skip")])
            fig.update_xaxes(range=[-lim2, lim2])
            fig.update_yaxes(range=[-lim2, lim2])
            st.plotly_chart(eje(estilo(fig, 330, "closest"), "corrección del Ridge (°C)",
                                "corrección de LightGBM (°C)"), width="stretch", config=SIN_BARRA)
            st.markdown(f'<p class="nota">Correlación entre las dos correcciones: {num(r)}.</p>',
                        unsafe_allow_html=True)

    st.markdown("### ¿Se captó el sesgo sistemático?")
    c1, c2 = st.columns(2, gap="small")
    with c1, st.container(border=True):
        tarjeta("Sesgo que queda", "Amplitud del ciclo del error medio por hora y desvío del error medio entre "
                "estaciones, en °C (pliegues de reporte, sin filtros).")
        SR = R["sesgo_restante"].assign(modelo=lambda d: d.modelo.map(NOMBRE)).rename(columns={
            "amplitud_hora": "amplitud por hora", "desvio_entre_estaciones": "desvío entre estaciones",
            "error_medio": "error medio"})
        tabla(SR)
        figura(FIG / "interp_sesgo_hora.png", "Error medio por hora solar: el ciclo del NWS y lo que queda.")
    with c2, st.container(border=True):
        tarjeta("Cuánto del error se anticipa", "Correlación entre la corrección y el error del NWS, por horizonte. "
                "El último error solo (err_ult) es la referencia de persistencia.")
        Te = R["techo"]
        fig = go.Figure()
        for col, nom, c, d in (("err_ult", "último error (err_ult)", GRIS, "dot"), ("lineal", "Ridge", AZUL, "solid"),
                               ("lgbm", "LightGBM", AMBAR, "solid")):
            fig.add_trace(go.Scatter(x=Te.horizonte, y=Te[col], name=nom, mode="lines+markers",
                                     line=dict(color=c, width=2, dash=d),
                                     marker=dict(size=8, line=dict(width=2, color=BLANCO))))
        st.plotly_chart(eje(estilo(fig, 300), "horizonte (h)", "correlación con el error"), width="stretch",
                        config=SIN_BARRA)
        CB = R["cobertura_global"].set_index("modelo").cobertura
        st.markdown(f'<p class="nota">Banda P10–P90 (calibrada en los pliegues 1–2): cubre {num(100 * CB["lineal"], 1)} % '
                    f'con el Ridge y {num(100 * CB["lgbm"], 1)} % con LightGBM, contra el 80 % prometido. Las dos son '
                    f'algo optimistas.</p>', unsafe_allow_html=True)

    st.markdown("### Balance")
    tabla(pd.DataFrame([
        ["Error (MAE reporte)", f"{num(li, 3)} °C", f"{num(lg, 3)} °C", "LightGBM, por 0,05 °C (p = 0,0002)"],
        ["Mejora sobre el NWS", f"{num(100 * (b0 - li) / b0, 1)} %", f"{num(100 * (b0 - lg) / b0, 1)} %", "LightGBM"],
        ["Variables", "5 familias", "8 familias (51 variables)", "Ridge: menos datos que mantener"],
        ["Hiperparámetros", "1 (α)", "3 elegidos + 6 fijos", "Ridge"],
        ["Cómo se explica", "coeficientes y curvas en °C", "SHAP (TreeSHAP exacto, por fila)", "Ridge"],
        ["Faltantes", "imputación + indicador", "nativo", "LightGBM"],
        ["Banda de confianza", f"{num(100 * CB['lineal'], 1)} % de cobertura", f"{num(100 * CB['lgbm'], 1)} % de cobertura",
         "LightGBM, levemente"],
        ["Entrenamiento por pliegue", "≈ 15 s (1 M filas)", "≈ 30 s (2 M filas)", "Ridge"],
    ], columns=["criterio", "Ridge (Gram)", "LightGBM", "a favor de"]))


# ======================================================================== 5. MONITOREO (vista original)
with t_mon:
    t1, t2 = st.columns([3, 1], vertical_alignment="bottom")
    hallazgo = t1.empty()
    f = st.columns([1.3, 1.2, 1.2, 1.6, 1.6])
    modelo = f[0].selectbox("Corrección", list(MODELOS), format_func=MODELOS.get, key="mo_modelo")
    h_rng = f[1].slider("Horizonte (h)", 1, 48, (1, 48), key="mo_h")
    hs_rng = f[2].slider("Hora solar local", 0, 23, (0, 23), key="mo_hs")
    conds = f[3].multiselect("Condición del tiempo", list(CONDICIONES), format_func=CONDICIONES.get,
                             placeholder="Todas", key="mo_cond")
    estados = f[4].multiselect("Estados", sorted(D["estado"].dropna().unique()), placeholder="Todos", key="mo_est")
    CM, BM = COLOR[modelo], BANDA[modelo]

    m = D["horizonte"].between(*h_rng).to_numpy() & D["hora_solar"].between(*hs_rng).to_numpy()
    if conds:
        m &= D["condicion"].isin(conds).to_numpy()
    if estados:
        m &= D["estado"].isin(estados).to_numpy()
    F = D[m]

    if F.empty:
        hallazgo.markdown('<p class="hallazgo">Ninguna fila cumple estos filtros. Ampliá el rango de horas o '
                          'quitá alguna condición.</p>', unsafe_allow_html=True)
    else:
        r_nws, r_cor = F["e_nws"], F[f"e_{modelo}"]
        e_nws, e_cor = mae(r_nws), mae(r_cor)
        mejora = 100 * (e_nws - e_cor) / e_nws
        dentro = ((F["obs"] >= F[modelo] + F[f"{modelo}_q10"]) & (F["obs"] <= F[modelo] + F[f"{modelo}_q90"])).mean()
        ancho_med = float((F[f"{modelo}_q90"] - F[f"{modelo}_q10"]).median())

        por_cond = F.assign(a=np.abs(r_nws), b=np.abs(r_cor)).groupby("condicion", observed=True)[["a", "b"]].mean()
        por_cond = por_cond[F["condicion"].value_counts().reindex(por_cond.index) >= 5000]
        peor = por_cond["b"].idxmax() if len(por_cond) > 1 else None
        texto = f"La corrección baja el error medio de {num(e_nws)} a {num(e_cor)} °C ({num(mejora, 1)} % menos)."
        if peor:
            texto += (f" Donde menos se puede confiar es con {CONDICIONES[peor].lower()}: "
                      f"queda en {num(por_cond.loc[peor, 'b'])} °C.")
        hallazgo.markdown(f'<p class="hallazgo">{texto}</p>', unsafe_allow_html=True)
        cols_csv = ["pliegue", "station_id", "target_time_utc", "horizonte", "hora_solar", "condicion", "nws", "obs",
                    "lineal", "lgbm", "historia", "estado"]
        t2.download_button("Descargar filas filtradas", F[cols_csv].head(50_000).to_csv(index=False),
                           file_name="pronostico_filtrado.csv", type="primary", width="stretch",
                           help="Hasta 50.000 filas, en CSV")

        # estación y horizonte de la serie: se eligen en la columna lateral; si la estación elegida quedó fuera de
        # los filtros, se toma la que más datos tiene
        por_est = (F.assign(a=np.abs(r_nws), b=np.abs(r_cor)).groupby("station_id", observed=True)[["a", "b"]]
                   .mean().join(E[["estado", "lat", "lon"]]))
        por_est["mejora"] = 100 * (por_est["a"] - por_est["b"]) / por_est["a"]
        if st.session_state.get("estacion") not in por_est.index:
            st.session_state["estacion"] = F["station_id"].value_counts().idxmax()
        st.session_state.setdefault("h_serie", 24)
        est_sel, h_sel = st.session_state["estacion"], st.session_state["h_serie"]

        main, rail = st.columns([2.6, 1], gap="medium")
        with main:
            kpis([("Error del NWS", f"{num(e_nws)} °C", f"MAE, {num(len(F), 0)} pronósticos", NWS),
                  ("Error corregido", f"{num(e_cor)} °C", f"MAE, {MODELOS[modelo].split(' (')[0]}", CM),
                  ("Mejora", f"{num(mejora, 1)} % <span style='font-size:14px;color:"
                   f"{POSITIVO if mejora > 0 else NEGATIVO}'>{'▲' if mejora > 0 else '▼'}</span>",
                   "frente al NWS crudo, mismas filas", None),
                  ("Dentro de la banda", f"{num(100 * dentro, 0)} %", f"esperado 80 % · ancho típico {num(ancho_med, 1)} °C",
                   None)])
            st.write("")
            with st.container(border=True):
                S = D[(D["station_id"] == est_sel) & (D["horizonte"] == h_sel)].set_index("target_time_utc").sort_index()
                S = S[~S.index.duplicated()]
                S = S.reindex(pd.date_range(S.index.min(), S.index.max(), freq="h")).rename_axis("target_time_utc")
                S = S.reset_index()                   # horas sin dato quedan vacías: la línea se corta, no se inventa
                nombre = f"{est_sel} ({E.loc[est_sel, 'estado']})" if est_sel in E.index else est_sel
                tarjeta(f"Observado, NWS y corregido · estación {nombre}, pronóstico a {h_sel} h",
                        "La banda es el rango donde cayó el 80 % de los errores corregidos en la calibración, para "
                        "ese horizonte y condición del tiempo.")
                fig = go.Figure()
                t = S["target_time_utc"]
                alto_b, bajo_b = S[modelo] + S[f"{modelo}_q90"], S[modelo] + S[f"{modelo}_q10"]
                tramo_id = alto_b.isna().cumsum()              # un polígono por tramo sin huecos
                for i, (_, idx) in enumerate(alto_b.dropna().groupby(tramo_id[alto_b.notna()]).groups.items()):
                    xs = pd.concat([t[idx], t[idx][::-1]])
                    ys = pd.concat([alto_b[idx], bajo_b[idx][::-1]])
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=BM, line=dict(width=0),
                                             hoverinfo="skip", name="Banda de confianza", showlegend=i == 0,
                                             legendgroup="banda"))
                fig.add_trace(go.Scatter(x=t, y=S["nws"], name="Pronóstico NWS", line=dict(color=NWS, width=1.5)))
                fig.add_trace(go.Scatter(x=t, y=S[modelo], name=f"Corregido ({NOMBRE[modelo]})",
                                         line=dict(color=CM, width=2)))
                fig.add_trace(go.Scatter(x=t, y=S["obs"], name="Observado", mode="markers",
                                         marker=dict(color=TINTA, size=4)))
                st.plotly_chart(eje(estilo(fig, 300), None, "°C"), width="stretch", config=SIN_BARRA)

            c1, c2 = st.columns(2, gap="small")
            with c1, st.container(border=True):
                tarjeta("Error según el horizonte", "MAE en °C. Más lejos, menos confianza.")
                g = F.assign(a=np.abs(r_nws), b=np.abs(r_cor)).groupby("horizonte")[["a", "b"]].mean()
                fig = go.Figure([go.Scatter(x=g.index, y=g["a"], name="NWS", line=dict(color=NWS, width=1.5)),
                                 go.Scatter(x=g.index, y=g["b"], name="Corregido", line=dict(color=CM, width=2))])
                st.plotly_chart(eje(estilo(fig, 220), "horizonte (h)"), width="stretch", config=SIN_BARRA)
            with c2, st.container(border=True):
                tarjeta("Error según la hora del día", "MAE en °C por hora solar local.")
                g = F.assign(a=np.abs(r_nws), b=np.abs(r_cor)).groupby("hora_solar")[["a", "b"]].mean()
                fig = go.Figure([go.Scatter(x=g.index, y=g["a"], name="NWS", line=dict(color=NWS, width=1.5)),
                                 go.Scatter(x=g.index, y=g["b"], name="Corregido", line=dict(color=CM, width=2))])
                fig.update_xaxes(dtick=3)
                st.plotly_chart(eje(estilo(fig, 220), "hora solar"), width="stretch", config=SIN_BARRA)

        with rail:
            tarjeta("Estación")
            st.selectbox("Estación", por_est.sort_values("b").index.tolist(), key="estacion",
                         label_visibility="collapsed",
                         format_func=lambda s: f"{s} · {por_est.loc[s, 'estado']} · error corregido "
                                               f"{num(por_est.loc[s, 'b'])} °C")
            st.select_slider("Horizonte de la serie", options=[1, 3, 6, 12, 24, 36, 48], key="h_serie")

            st.markdown('<p class="card-h" style="margin-top:16px">Mejora por estación</p>'
                        '<p class="card-s">Color más intenso, más mejora; rojo, la corrección empeora.</p>',
                        unsafe_allow_html=True)
            with st.container(border=True):
                lim = float(np.nanpercentile(np.abs(por_est["mejora"]), 95))
                fig = go.Figure(go.Scattergeo(
                    lon=por_est["lon"], lat=por_est["lat"], text=por_est.index, mode="markers",
                    marker=dict(size=6, color=por_est["mejora"], cmin=-lim, cmax=lim,
                                line=dict(width=0.5, color=BLANCO),
                                colorscale=[[0, NEGATIVO], [0.5, "#d4d4d4"], [1, CM]]),
                    hovertemplate="%{text}<br>mejora %{marker.color:.1f} %<extra></extra>"))
                fig.update_geos(scope="north america", projection_type="mercator", showland=True,
                                landcolor="#f7f7f7", showsubunits=True, subunitcolor=BORDE, countrycolor=BORDE,
                                showlakes=False, lonaxis_range=[-92, -66], lataxis_range=[24, 48], bgcolor=BLANCO)
                st.plotly_chart(estilo(fig, 300, "closest"), width="stretch", config=SIN_BARRA)

            tarjeta("Donde menos ayuda", "Estaciones con menor mejora en la selección.")
            filas = "".join(f'<div class="fila"><span>{s} <span class="g">{r.estado}</span></span>'
                            f'<span>{num(r.mejora, 1)} % <span class="g">· {num(r.b)} °C</span></span></div>'
                            for s, r in por_est.sort_values("mejora").head(5).iterrows())
            st.markdown(filas, unsafe_allow_html=True)


# ======================================================================== 6. VARIABLES
with t_var:
    st.markdown('<p class="hallazgo">Las 8 familias se calculan solo con lo que se sabía al emitir cada pronóstico '
                '(control automático: 0 violaciones de causalidad en 32,6 M de filas). El Ridge usa 5 familias y '
                'LightGBM las 8.</p>', unsafe_allow_html=True)
    st.write("")
    tabla(pd.DataFrame([{"familia": f, "Ridge": "sí" if f in FAM_RIDGE else "no", "LightGBM": "sí",
                         "qué mide": FAMILIAS[f][0], "variables": FAMILIAS[f][1]} for f in FAMILIAS]))
    c1, c2 = st.columns([1.1, 1], gap="small")
    with c1, st.container(border=True):
        tarjeta("Correlación de cada variable con el error del NWS", "Spearman, días de exploración. Elegí el "
                "horizonte; la barra muestra el valor absoluto y el texto, el signo.")
        C = X["correlaciones_con_el_error"]
        hcol = st.radio("Horizonte", ["total", "h1", "h6", "h24", "h48"], horizontal=True, key="var_h",
                        format_func={"total": "todos", "h1": "1 h", "h6": "6 h", "h24": "24 h", "h48": "48 h"}.get)
        fams = st.multiselect("Familias", list(FAMILIAS), default=["base", "pendiente", "rezagos", "sesgo_ewma"],
                              key="var_f")
        Cc = C[C.familia.isin(fams)].dropna(subset=[hcol]).assign(abs=lambda d: d[hcol].abs()).sort_values("abs")
        fig = go.Figure(go.Bar(y=Cc.variable, x=Cc["abs"], orientation="h",
                               marker=dict(color=[AZUL if f in FAM_RIDGE else GRIS for f in Cc.familia],
                                           cornerradius=4),
                               text=[f"{'+' if x >= 0 else '−'} · {f}" for x, f in zip(Cc[hcol], Cc.familia)],
                               textposition="outside", textfont=dict(color=GRIS, size=10),
                               customdata=Cc[hcol], hovertemplate="%{y}: %{customdata:.3f}<extra></extra>"))
        fig.update_xaxes(range=[0, max(Cc["abs"].max() * 1.35, 0.05)])
        st.plotly_chart(eje(estilo(fig, max(300, 18 * len(Cc)), "closest"), "|correlación de Spearman|"),
                        width="stretch", config=SIN_BARRA)
        st.markdown('<p class="nota">Azul = la familia está en el Ridge; gris = solo en LightGBM.</p>',
                    unsafe_allow_html=True)
    with c2, st.container(border=True):
        tarjeta("Redundancia entre familias", "Mayor correlación de Spearman entre una variable de cada familia "
                "(diagonal: dentro de la misma familia). Descriptiva: no decide.")
        Rd = X["redundancia_entre_familias"]
        fig = go.Figure(go.Heatmap(z=Rd.values, x=Rd.columns, y=Rd.index, zmin=0, zmax=1,
                                   colorscale=[[0, "#f4f7fc"], [1, AZUL]],
                                   text=np.vectorize(lambda x: "" if np.isnan(x) else num(x))(Rd.values),
                                   texttemplate="%{text}", textfont=dict(size=10),
                                   hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>", xgap=2, ygap=2,
                                   colorbar=dict(thickness=10, tickfont=dict(size=10, color=GRIS))))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(estilo(fig, 420, "closest"), width="stretch", config=SIN_BARRA)
        st.markdown('<p class="nota">Rezagos y rezagos extra (0,87) y rezagos y vecinos (0,69) se repiten: por eso '
                    'las familias experimentales no le suman al Ridge.</p>', unsafe_allow_html=True)

    st.markdown("### Exploración por familia")
    figs = {"Base": "exploracion_base.png", "Base: condición y noche": "exploracion_base_condicion_noche.png",
            "Pendiente": "exploracion_pendiente.png", "Rezagos": "exploracion_rezagos.png",
            "Sesgo EWMA": "exploracion_sesgo_ewma.png", "Vecinos": "exploracion_vecinos.png",
            "Revisión": "exploracion_revision.png", "Rezagos extra": "exploracion_rezagos_extra.png",
            "Dinámica": "exploracion_dinamica.png", "Mapa de correlaciones": "mapa_correlaciones.png"}
    elegida = st.selectbox("Figura", list(figs), key="var_fig")
    figura(FIG_EXP / figs[elegida], f"{elegida} (notebooks/experimentos.ipynb).")


# ======================================================================== 8. DATOS Y VALIDACIÓN
with t_val:
    PL, EX, BL = R["pliegues"], R["dias_excluidos"], R["baselines"]
    st.markdown('<p class="hallazgo">De dónde salen los números: qué días entran, cómo se separa el entrenamiento de '
                'la prueba (validación cruzada walk-forward) y las correcciones simples que los modelos tienen que '
                'superar.</p>', unsafe_allow_html=True)
    st.write("")
    kpis([("Días con datos", "55", "10/09 → 09/11/2026 (CSV diarios del TP1)", None),
          ("Días usables", str(55 - len(EX)), f"{len(EX)} excluidos por la regla automática", None),
          ("Pliegues", str(len(PL)), f"{(PL.uso == 'elige').sum()} eligen hiperparámetros · "
                                      f"{(PL.uso == 'reporta').sum()} reportan", None),
          ("Filas pronóstico–observación", "32,6 M", "519 estaciones × hora × horizonte 1–48 h", None)])
    st.write("")

    with st.container(border=True):
        tarjeta("Validación walk-forward con pliegues anclados", "Cada pliegue entrena con todo lo anterior (ventana "
                "expansiva), deja 2–3 días de separación y prueba con los 5 días usables siguientes. Los pliegues 1–2 "
                "eligen los hiperparámetros (una vez, quedan congelados); los 3–5 dan la métrica del informe.")
        a = lambda d: pd.Timestamp(f"2026-{d[3:5]}-{d[0:2]}")
        fig = go.Figure()
        for _, r in PL.iterrows():
            y = f"pliegue {r.pliegue} · {r.uso}"
            ini, fin_e, ip, fp = a(r.entrena_desde), a(r.entrena_hasta), a(r.prueba_desde), a(r.prueba_hasta)
            fig.add_trace(go.Bar(y=[y], x=[((fin_e - ini).days + 1) * 86_400_000], base=[ini.strftime("%Y-%m-%d")],
                                 orientation="h", marker=dict(color="#d4d4d4", cornerradius=4), name="entrenamiento",
                                 showlegend=r.pliegue == 1, legendgroup="e",
                                 hovertemplate=f"entrena {r.entrena_desde} → {r.entrena_hasta} ({r.dias_entrena} días)"
                                               "<extra></extra>"))
            fig.add_trace(go.Bar(y=[y], x=[((fp - ip).days + 1) * 86_400_000], base=[ip.strftime("%Y-%m-%d")], orientation="h",
                                 marker=dict(color=GRIS if r.uso == "elige" else TINTA, cornerradius=4),
                                 name="prueba (elige)" if r.uso == "elige" else "prueba (reporta)",
                                 showlegend=r.pliegue in (1, 3), legendgroup=r.uso,
                                 hovertemplate=f"prueba {r.prueba_desde} → {r.prueba_hasta} ({r.prueba})"
                                               "<extra></extra>"))
        for d in EX.dia:
            x = pd.Timestamp(f"{d[6:]}-{d[3:5]}-{d[0:2]}")
            fig.add_vrect(x0=x, x1=x + pd.Timedelta(days=1), fillcolor=NEGATIVO, opacity=0.08, line_width=0)
        fig.update_layout(barmode="overlay", bargap=0.35)
        fig.update_xaxes(type="date", dtick=7 * 86400000, tickformat="%d/%m")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(estilo(fig, 280, "closest"), width="stretch", config=SIN_BARRA)
        st.markdown('<p class="nota">Franja roja: días excluidos. Entre los bloques también faltan días sin archivo '
                    '(15/10, 27–28/10, 30/10–01/11). El pliegue 4 saltea esos días para completar sus 5 días de '
                    'prueba.</p>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.2], gap="small")
    with c1, st.container(border=True):
        tarjeta("Días excluidos", "Regla: se excluye un día si su % de filas sin observación supera en más de 10 "
                "puntos la mediana de los 14 días válidos anteriores. Causa confirmada: falla de publicación del NWS.")
        tabla(EX.rename(columns={"dia": "día", "sin_observacion_pct": "sin observación %",
                                 "referencia_pct": "referencia %", "limite_pct": "límite %"}))
        st.markdown('<p class="nota">Además, 11 días tienen «historia incompleta» (les falta alguno de los 3 días '
                    'previos): no se excluyen, pero se miden aparte porque sus rezagos quedan vacíos.</p>',
                    unsafe_allow_html=True)
    with c2, st.container(border=True):
        tarjeta("MAE por pliegue de reporte", "°C, ponderado por filas a partir del MAE diario. Si un modelo se "
                "sobreajustara a la elección, su ventaja caería en estos pliegues.")
        Dd = R["mae_por_dia"]
        Pp = Dd.groupby("pliegue").apply(lambda g: pd.Series({m: np.average(g[m], weights=g.n)
                                                              for m in ("B0", "B2", "lineal", "lgbm")}),
                                         include_groups=False)
        fig = go.Figure()
        for m in ("B0", "B2", "lineal", "lgbm"):
            fig.add_trace(go.Bar(x=[f"pliegue {i}" for i in Pp.index], y=Pp[m], name=NOMBRE[m],
                                 marker=dict(color=COLOR[m], cornerradius=4),
                                 hovertemplate="%{x}: %{y:.3f} °C<extra>" + NOMBRE[m] + "</extra>"))
        fig.update_layout(bargap=0.3, bargroupgap=0.08)
        fig.update_yaxes(range=[1.0, float(Pp.values.max()) * 1.05])
        st.plotly_chart(eje(estilo(fig, 300), None, "MAE (°C)"), width="stretch", config=SIN_BARRA)
        st.markdown('<p class="nota">El eje empieza en 1 °C para que se vean las diferencias. El orden NWS > B2 > '
                    'Ridge > LightGBM se mantiene en los tres pliegues.</p>', unsafe_allow_html=True)

    st.markdown("### Baselines: la vara mínima")
    st.markdown('<p class="card-s">B0 = NWS sin corregir · B1 = error medio global del entrenamiento · B2 = error medio '
                'de la estación a esa hora solar · B3 = último error conocido (persistencia). Pliegues 1–5, mismas '
                'filas para los cuatro (donde B2 y B3 existen).</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1], gap="small")
    with c1, st.container(border=True):
        seg = st.radio("Días de prueba", ["todas", "completa", "incompleta"], horizontal=True, key="val_seg",
                       format_func={"todas": "Todos", "completa": "Historia completa",
                                    "incompleta": "Historia incompleta"}.get)
        G = BL[(BL.segmento == seg) & (BL.metrica == "MAE") & (BL.horizonte != "todos")].copy()
        G["h"] = G.horizonte.astype(int)
        col_b = {"B0": NWS, "B1": "#a3a3a3", "B2": B2C, "B3": TINTA}
        dash_b = {"B0": "solid", "B1": "dash", "B2": "dot", "B3": "dashdot"}
        fig = go.Figure()
        for b in ("B0", "B1", "B2", "B3"):
            g = G[G.baseline == b].sort_values("h")
            fig.add_trace(go.Scatter(x=g.h, y=g.valor, name=b, mode="lines+markers",
                                     line=dict(color=col_b[b], width=2, dash=dash_b[b]),
                                     marker=dict(size=8, line=dict(width=2, color=BLANCO))))
        st.plotly_chart(eje(estilo(fig, 300), "horizonte (h)", "MAE (°C)"), width="stretch", config=SIN_BARRA)
    with c2, st.container(border=True):
        T = BL[(BL.segmento == seg) & (BL.horizonte == "todos")].pivot_table(index="metrica", columns="baseline",
                                                                            values="valor")
        T = T.reindex(["MAE", "RMSE", "sesgo"]).reset_index().rename(columns={"metrica": "todas las horas"})
        tabla(T, column_config={b: st.column_config.NumberColumn(format="%.3f") for b in ("B0", "B1", "B2", "B3")})
        st.markdown('<p class="nota">B2 es el mejor baseline en casi todos los horizontes: hay sesgo por estación y '
                    'hora que se puede aprender. B1 no mejora a B0 (el sesgo global no es estable entre semanas) y la '
                    'persistencia (B3) solo compite a 1 h. Los modelos se comparan contra B0 y B2.</p>',
                    unsafe_allow_html=True)
