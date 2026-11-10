"""Dashboard del TP3 (Etapa C): confianza del pronóstico de temperatura del NWS y su corrección.

Pregunta de la pantalla: ¿cuánto confiar en el pronóstico y cuánto lo mejora la corrección?
Datos: días de prueba de los pliegues de reporte (fuera de muestra), preparados por src/preparar_dashboard.py.
Estilo: A, «libro contable acromático» (taste-skill): la interfaz sin color; el azul marca solo el pronóstico
corregido y su banda de confianza.

Uso (desde TP3/):  .venv/bin/streamlit run app/dashboard.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "dashboard"

# tokens del estilo A
TINTA, GRIS, BORDE, REGLA, FONDO, BLANCO = "#0a0a0a", "#6b6b6b", "#e6e6e6", "#f0f0f0", "#f2f4f7", "#ffffff"
AZUL, AZUL_BANDA, POSITIVO, NEGATIVO, NWS = "#1d5fbf", "rgba(29,95,191,0.14)", "#2e7d32", "#c4242a", "#8a8a8a"
MODELOS = {"lineal": "Ridge (interpretable)", "lgbm": "LightGBM (mínimo error)"}
CONDICIONES = {"despejado": "Despejado", "parcial": "Parcialmente nublado", "nublado": "Nublado", "lluvia": "Lluvia",
               "tormenta": "Tormenta", "niebla": "Niebla", "nieve": "Nieve", "otro": "Otro"}

st.set_page_config(page_title="Confianza del pronóstico · NWS Este", layout="wide")
st.markdown(f"""<style>
html, body, [class*="css"] {{ font-variant-numeric: tabular-nums; }}
.block-container {{ padding-top: 3rem; max-width: 1440px; }}
h1 {{ font-size: 24px !important; font-weight: 700 !important; margin: 0 !important; padding: 0 !important; }}
.sub {{ color: {GRIS}; font-size: 14px; margin: 2px 0 0; }}
.hallazgo {{ color: {TINTA}; font-size: 14px; margin: 6px 0 0; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); background: {BLANCO}; border: 1px solid {BORDE};
        border-radius: 8px; }}
.kpi {{ padding: 14px 16px; border-right: 1px solid {BORDE}; }}
.kpi:last-child {{ border-right: none; }}
.kpi .l {{ color: {GRIS}; font-size: 14px; }}
.kpi .v {{ font-size: 30px; font-weight: 500; line-height: 1.25; }}
.kpi .c {{ color: {GRIS}; font-size: 12px; }}
.card-h {{ font-size: 16px; font-weight: 500; margin: 0; }}
.card-s {{ color: {GRIS}; font-size: 14px; margin: 2px 0 8px; }}
.fila {{ display: flex; justify-content: space-between; padding: 11px 0; border-bottom: 1px solid {REGLA};
        font-size: 14px; }}
.fila .g {{ color: {GRIS}; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: {BLANCO}; }}
</style>""", unsafe_allow_html=True)


def num(x, d=2):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_resource
def cargar():
    D = pd.read_parquet(DATOS / "filas.parquet")
    E = pd.read_parquet(DATOS / "estaciones.parquet").set_index("station_id")
    D = D.join(E["estado"], on="station_id")
    return D, E


def mae(r):
    return float(np.abs(r).mean()) if len(r) else np.nan


def estilo(fig, alto):
    fig.update_layout(height=alto, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor=BLANCO, plot_bgcolor=BLANCO,
                      font=dict(family="Geist, Inter, sans-serif", size=12, color=TINTA), hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=12, color=GRIS)))
    fig.update_xaxes(showgrid=False, linecolor=BORDE, tickfont=dict(color=GRIS, size=11))
    fig.update_yaxes(gridcolor=REGLA, zeroline=False, tickfont=dict(color=GRIS, size=11))
    return fig


D, E = cargar()

# ---------------------------------------------------------------- título y alcance
t1, t2 = st.columns([3, 1], vertical_alignment="bottom")
dias = D["target_time_utc"].dt.tz_convert("UTC").dt.normalize()
t1.markdown("# Confianza del pronóstico de temperatura")
t1.markdown(f'<p class="sub">NWS en {D.station_id.nunique()} estaciones del Este de EE.UU. · '
            f'{dias.nunique()} días de prueba entre el {dias.min():%d/%m} y el {dias.max():%d/%m/%Y}, '
            f'que los modelos no vieron al entrenar</p>', unsafe_allow_html=True)
hallazgo = t1.empty()

f = st.columns([1.3, 1.2, 1.2, 1.6, 1.6])
modelo = f[0].selectbox("Corrección", list(MODELOS), format_func=MODELOS.get)
h_rng = f[1].slider("Horizonte (h)", 1, 48, (1, 48))
hs_rng = f[2].slider("Hora solar local", 0, 23, (0, 23))
conds = f[3].multiselect("Condición del tiempo", list(CONDICIONES), format_func=CONDICIONES.get,
                         placeholder="Todas")
estados = f[4].multiselect("Estados", sorted(D["estado"].dropna().unique()), placeholder="Todos")

m = D["horizonte"].between(*h_rng).to_numpy() & D["hora_solar"].between(*hs_rng).to_numpy()
if conds:
    m &= D["condicion"].isin(conds).to_numpy()
if estados:
    m &= D["estado"].isin(estados).to_numpy()
F = D[m]

if F.empty:
    hallazgo.markdown('<p class="hallazgo">Ninguna fila cumple estos filtros. Ampliá el rango de horas o '
                      'quitá alguna condición.</p>', unsafe_allow_html=True)
    st.stop()

r_nws, r_cor = F["obs"] - F["nws"], F["obs"] - F[modelo]
e_nws, e_cor = mae(r_nws), mae(r_cor)
mejora = 100 * (e_nws - e_cor) / e_nws
dentro = ((F["obs"] >= F[modelo] + F[f"{modelo}_q10"]) & (F["obs"] <= F[modelo] + F[f"{modelo}_q90"])).mean()
ancho_med = float((F[f"{modelo}_q90"] - F[f"{modelo}_q10"]).median())

por_cond = F.assign(a=np.abs(r_nws), b=np.abs(r_cor)).groupby("condicion", observed=True)[["a", "b"]].mean()
por_cond = por_cond[F["condicion"].value_counts().reindex(por_cond.index) >= 5000]
peor = por_cond["b"].idxmax() if len(por_cond) > 1 else None
texto = (f"La corrección baja el error medio de {num(e_nws)} a {num(e_cor)} °C ({num(mejora, 1)} % menos).")
if peor:
    texto += (f" Donde menos se puede confiar es con {CONDICIONES[peor].lower()}: "
              f"queda en {num(por_cond.loc[peor, 'b'])} °C.")
hallazgo.markdown(f'<p class="hallazgo">{texto}</p>', unsafe_allow_html=True)
t2.download_button("Descargar filas filtradas", F.drop(columns=["tramo"]).head(50_000).to_csv(index=False),
                   file_name="pronostico_filtrado.csv", type="primary", width="stretch",
                   help="Hasta 50.000 filas, en CSV")

# estación y horizonte de la serie: se eligen en la columna lateral; si la estación elegida quedó fuera de los
# filtros, se toma la que más datos tiene
por_est = (F.assign(a=np.abs(r_nws), b=np.abs(r_cor)).groupby("station_id", observed=True)[["a", "b"]]
           .mean().join(E[["estado", "lat", "lon"]]))
por_est["mejora"] = 100 * (por_est["a"] - por_est["b"]) / por_est["a"]
if st.session_state.get("estacion") not in por_est.index:
    st.session_state["estacion"] = F["station_id"].value_counts().idxmax()
st.session_state.setdefault("h_serie", 24)
est_sel, h_sel = st.session_state["estacion"], st.session_state["h_serie"]

# ---------------------------------------------------------------- cuerpo: columna principal y lateral
main, rail = st.columns([2.6, 1], gap="medium")

with main:
    st.markdown(f"""<div class="kpis">
<div class="kpi"><div class="l">Error del NWS</div><div class="v">{num(e_nws)} °C</div>
<div class="c">MAE, {num(len(F), 0)} pronósticos</div></div>
<div class="kpi"><div class="l">Error corregido</div><div class="v">{num(e_cor)} °C</div>
<div class="c">MAE, {MODELOS[modelo].split(' (')[0]}</div></div>
<div class="kpi"><div class="l">Mejora</div><div class="v">{num(mejora, 1)} %
<span style="font-size:14px;color:{POSITIVO if mejora > 0 else NEGATIVO}">{'▲' if mejora > 0 else '▼'}</span></div>
<div class="c">frente al NWS crudo, mismas filas</div></div>
<div class="kpi"><div class="l">Dentro de la banda</div><div class="v">{num(100 * dentro, 0)} %</div>
<div class="c">esperado 80 % · ancho típico {num(ancho_med, 1)} °C</div></div>
</div>""", unsafe_allow_html=True)

    st.write("")
    with st.container(border=True):
        S = D[(D["station_id"] == est_sel) & (D["horizonte"] == h_sel)].set_index("target_time_utc").sort_index()
        S = S[~S.index.duplicated()]
        S = S.reindex(pd.date_range(S.index.min(), S.index.max(), freq="h")).rename_axis("target_time_utc")
        S = S.reset_index()                       # horas sin dato quedan vacías: la línea se corta, no se inventa
        nombre = f"{est_sel} ({E.loc[est_sel, 'estado']})" if est_sel in E.index else est_sel
        st.markdown(f'<p class="card-h">Observado, NWS y corregido · estación {nombre}, pronóstico a {h_sel} h</p>'
                    f'<p class="card-s">La banda azul es el rango donde cayó el 80 % de los errores corregidos en la '
                    f'calibración, para ese horizonte y condición del tiempo.</p>', unsafe_allow_html=True)
        fig = go.Figure()
        t = S["target_time_utc"]
        alto_b, bajo_b = S[modelo] + S[f"{modelo}_q90"], S[modelo] + S[f"{modelo}_q10"]
        tramo_id = alto_b.isna().cumsum()                  # un polígono por tramo sin huecos
        for i, (_, idx) in enumerate(alto_b.dropna().groupby(tramo_id[alto_b.notna()]).groups.items()):
            xs = pd.concat([t[idx], t[idx][::-1]])
            ys = pd.concat([alto_b[idx], bajo_b[idx][::-1]])
            fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor=AZUL_BANDA, line=dict(width=0),
                                     hoverinfo="skip", name="Banda de confianza", showlegend=i == 0,
                                     legendgroup="banda"))
        fig.add_trace(go.Scatter(x=t, y=S["nws"], name="Pronóstico NWS", line=dict(color=NWS, width=1.5)))
        fig.add_trace(go.Scatter(x=t, y=S[modelo], name="Corregido", line=dict(color=AZUL, width=2)))
        fig.add_trace(go.Scatter(x=t, y=S["obs"], name="Observado", mode="markers",
                                 marker=dict(color=TINTA, size=4)))
        fig.update_yaxes(title=dict(text="°C", font=dict(color=GRIS, size=11)))
        st.plotly_chart(estilo(fig, 300), width="stretch", config={"displayModeBar": False})

    c1, c2 = st.columns(2, gap="small")
    with c1, st.container(border=True):
        st.markdown('<p class="card-h">Error según el horizonte</p><p class="card-s">MAE en °C. Más lejos, menos '
                    'confianza.</p>', unsafe_allow_html=True)
        g = F.assign(a=np.abs(r_nws), b=np.abs(r_cor)).groupby("horizonte")[["a", "b"]].mean()
        fig = go.Figure([go.Scatter(x=g.index, y=g["a"], name="NWS", line=dict(color=NWS, width=1.5)),
                         go.Scatter(x=g.index, y=g["b"], name="Corregido", line=dict(color=AZUL, width=2))])
        fig.update_xaxes(title=dict(text="horizonte (h)", font=dict(color=GRIS, size=11)))
        st.plotly_chart(estilo(fig, 220), width="stretch", config={"displayModeBar": False})
    with c2, st.container(border=True):
        st.markdown('<p class="card-h">Error según la hora del día</p><p class="card-s">MAE en °C por hora solar '
                    'local.</p>', unsafe_allow_html=True)
        g = F.assign(a=np.abs(r_nws), b=np.abs(r_cor)).groupby("hora_solar")[["a", "b"]].mean()
        fig = go.Figure([go.Scatter(x=g.index, y=g["a"], name="NWS", line=dict(color=NWS, width=1.5)),
                         go.Scatter(x=g.index, y=g["b"], name="Corregido", line=dict(color=AZUL, width=2))])
        fig.update_xaxes(title=dict(text="hora solar", font=dict(color=GRIS, size=11)), dtick=3)
        st.plotly_chart(estilo(fig, 220), width="stretch", config={"displayModeBar": False})

with rail:
    st.markdown('<p class="card-h">Estación</p>', unsafe_allow_html=True)
    st.selectbox("Estación", por_est.sort_values("b").index.tolist(), key="estacion", label_visibility="collapsed",
                 format_func=lambda s: f"{s} · {por_est.loc[s, 'estado']} · error corregido {num(por_est.loc[s, 'b'])} °C")
    st.select_slider("Horizonte de la serie", options=[1, 3, 6, 12, 24, 36, 48], key="h_serie")

    st.markdown('<p class="card-h" style="margin-top:16px">Mejora por estación</p>'
                '<p class="card-s">Azul más oscuro, más mejora; rojo, la corrección empeora.</p>',
                unsafe_allow_html=True)
    with st.container(border=True):
        lim = float(np.nanpercentile(np.abs(por_est["mejora"]), 95))
        fig = go.Figure(go.Scattergeo(
            lon=por_est["lon"], lat=por_est["lat"], text=por_est.index, mode="markers",
            marker=dict(size=6, color=por_est["mejora"], cmin=-lim, cmax=lim, line=dict(width=0.5, color=BLANCO),
                        colorscale=[[0, NEGATIVO], [0.5, "#d4d4d4"], [1, AZUL]]),
            hovertemplate="%{text}<br>mejora %{marker.color:.1f} %<extra></extra>"))
        fig.update_geos(scope="north america", projection_type="mercator", showland=True, landcolor="#f7f7f7",
                        showsubunits=True, subunitcolor=BORDE, countrycolor=BORDE, showlakes=False,
                        lonaxis_range=[-92, -66], lataxis_range=[24, 48], bgcolor=BLANCO)
        st.plotly_chart(estilo(fig, 300), width="stretch", config={"displayModeBar": False})

    st.markdown('<p class="card-h" style="margin-top:16px">Donde menos ayuda</p>'
                '<p class="card-s">Estaciones con menor mejora en la selección.</p>', unsafe_allow_html=True)
    filas = "".join(f'<div class="fila"><span>{s} <span class="g">{r.estado}</span></span>'
                    f'<span>{num(r.mejora, 1)} % <span class="g">· {num(r.b)} °C</span></span></div>'
                    for s, r in por_est.sort_values("mejora").head(5).iterrows())
    st.markdown(filas, unsafe_allow_html=True)
