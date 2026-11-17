"""Feature engineering del TP3 (familias 1 a 4 acordadas el 2/11/2026).

Todas las variables se calculan con lo que existía AL EMITIR el pronóstico; no hay nada
que "aprender" del entrenamiento, así que se generan una sola vez para todas las filas
(data/features/<dia>.parquet) y los pliegues solo eligen qué días usar.

  1. Base         pronóstico y sus variables, horizonte, hora solar (sen/cos), ubicación, elevación del
                  sol y noche (sol bajo el horizonte), anomalía de la temperatura pronosticada
  2. Pendiente    cuánto cambia la temperatura pronosticada alrededor de la hora objetivo
  3. Rezagos      últimos errores conocidos de la misma estación y el mismo horizonte
  4. Sesgo EWMA   promedio exponencial de días del error por estación y hora (tipo NBM)
  5. Vecinos      últimos errores conocidos de las K estaciones más cercanas (dentro de max_km). EXPERIMENTO:
                  se calcula y se verifica, pero los modelos del producto no la usan (solo con --vecinos)
  6. Revisión     cuánto cambió el pronóstico para la MISMA hora objetivo respecto del emitido k horas antes
                  (rev_kh = pronóstico actual − pronóstico con horizonte h+k). EXPERIMENTO: solo con --revision
  7. Rezagos extra (EXPERIMENTO, solo con --rezagos)
                  a) err_hace{k}h: error del mismo horizonte de la hora  objetivo − h − k  (k = 9..21 h antes de la emisión)
                  b) err1h_*: error del pronóstico A 1 h de la propia estación (lo más fresco: «cuánto venimos
                     errando hoy»), último conocido y medias de las últimas 2, 4 y 8 horas, misma fuente que err_ult
  8. Dinámica     (EXPERIMENTO, solo con --dinamica; misma fuente que err_ult salvo donde se indica)
                  a) tendencia: obs_tend3h (cambio OBSERVADO en las últimas 3 h), err1h_tend3h (cambio del error a 1 h
                     = tendencia observada − pronosticada) y salto_obs (pronóstico actual − última temperatura observada)
                  b) rocío: err_rocio_ult / err_rocio_1h = rocío observado − pronosticado (mismo horizonte / a 1 h).
                     El rocío observado sale de src/rocio_observado.py (data/obs_rocio.parquet)
                  c) err_std6 / err_std24: desvío del error (mismo horizonte) en las últimas 6 / 24 h
                  d) dispersion_emis: desvío del pronóstico para la MISMA hora T entre la emisión actual y las de
                     hasta 24 h antes (horizontes h..h+24): «dispersión de ensamble» de pobre
                  e) amplitud_pron: máximo − mínimo pronosticado en ±12 h alrededor de T, misma emisión

DÍAS EXCLUIDOS POR CALIDAD (src/calidad_dias.py): las features se calculan para todos los días. Las
observaciones REALES de un día excluido pueden servir de rezago a días posteriores (son correctas,
solo incompletas); lo que no hace es aparecer como fila de entrenamiento ni de prueba, porque los
pliegues salen de validacion.dias_disponibles(), que ya los quita.

REGLA DE CAUSALIDAD: la fuente más reciente que se usa es la hora  objetivo - horizonte - 2 h.
(La observación de la hora H puede llegar hasta H+59 min y el pronóstico se genera entre 60 min
antes y 20 min después de objetivo - horizonte.)  Solo se usan observaciones REALES.

Uso:  .venv/bin/python src/features.py
"""
from pathlib import Path
import glob
import json
import warnings

import numpy as np
import pandas as pd
import yaml

from formato import dec

RAIZ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((RAIZ / "config.yaml").read_text())
ALFAS = CFG["features"]["alfas_ewma"]
RETRASO = 2                                  # horas de seguridad (ver REGLA DE CAUSALIDAD)
NH = 48                                      # horizontes 1..48
DIAS_ANOM, MIN_DIAS_ANOM = 7, 3              # anomalía de temperatura: media de los pronósticos de 7 días previos
SOL_HORIZONTE = -0.833                       # elevación del sol al salir/ponerse (refracción + radio del disco)
VEC = CFG["features"]["vecinos"]
REVISION_H = (3, 6, 12, 24)                  # familia 6: horas entre la emisión actual y la anterior comparada
REZAGOS_LARGOS_H = (9, 12, 15, 18, 21)       # familia 7a: horas antes de la emisión
VENTANAS_1H = ((2, 1), (4, 2), (8, 4))       # familia 7b: (ventana en horas, mínimo de datos)
VENTANAS_STD = ((6, 3), (24, 8))             # familia 8c: (ventana en horas, mínimo de datos)
DISP_H, MIN_DISP = 24, 3                     # familia 8d: emisiones de hasta 24 h antes, al menos 3 valores
AMP_H, MIN_AMP = 12, 12                      # familia 8e: ±12 h alrededor de T, al menos 12 valores

COLS_LECTURA = ["station_id", "state", "latitude", "longitude", "forecast_generated_at_utc", "target_time_utc",
                "horizon_target_hours", "temperature_forecast_c", "relative_humidity_forecast_pct",
                "dewpoint_forecast_c", "wind_speed_forecast_kmh", "wind_direction_forecast",
                "precip_probability_forecast_pct", "short_forecast", "error_c", "obs_interpolada"]

RUMBOS = {"N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
          "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5}


def agrupar_condicion(texto):
    """34 descripciones del NWS -> 7 grupos. El orden importa: lo más severo primero."""
    t = str(texto).lower()
    for grupo, claves in (("tormenta", ("thunder", "t-storm", "hurricane", "tropical storm")), ("nieve", ("snow", "sleet", "freezing", "flurr", "ice")),
                          ("lluvia", ("rain", "shower", "drizzle")), ("niebla", ("fog", "haze", "mist", "smoke")),
                          ("parcial", ("partly",)), ("nublado", ("cloudy", "overcast")),
                          ("despejado", ("sunny", "clear"))):
        if any(c in t for c in claves):
            return grupo
    return "otro"


def k_dias(h):
    """Días hacia atrás para que  24*k >= h + RETRASO  (mismo horario de un día anterior ya conocido)."""
    return np.ceil((h + RETRASO) / 24).astype(int)


def construir_tensores():
    """Primera pasada: arma tensores densos [horizonte, estación, hora] con errores reales y pronósticos."""
    archivos = sorted(glob.glob(str(RAIZ / "data/pares/*.parquet")))
    dias = pd.DatetimeIndex([pd.to_datetime(Path(f).stem, format="%Y_%m_%d", utc=True) for f in archivos])
    estaciones = pd.Index(sorted(pd.read_parquet(RAIZ / "data/obs_horaria.parquet", columns=["station_id"])
                                 ["station_id"].astype(str).unique()))
    t0 = dias.min()
    nT = (int((dias.max() - t0) / pd.Timedelta(days=1)) + 1) * 24      # días completos
    ERR = np.full((NH, len(estaciones), nT), np.nan, dtype=np.float32)   # error REAL (obs - pron)
    TEMP = np.full((NH, len(estaciones), nT), np.nan, dtype=np.float32)  # temperatura pronosticada
    ROCP = np.full((NH, len(estaciones), nT), np.nan, dtype=np.float32)  # rocío pronosticado
    for f in archivos:
        d = pd.read_parquet(f, columns=["station_id", "target_time_utc", "horizon_target_hours",
                                        "temperature_forecast_c", "dewpoint_forecast_c", "error_c", "obs_interpolada"])
        s = estaciones.get_indexer(d["station_id"].astype(str))
        t = ((d["target_time_utc"] - t0) / pd.Timedelta(hours=1)).astype(int).to_numpy()
        h = d["horizon_target_hours"].to_numpy().astype(int) - 1
        TEMP[h, s, t] = d["temperature_forecast_c"].to_numpy()
        ROCP[h, s, t] = d["dewpoint_forecast_c"].to_numpy()
        real = d["error_c"].notna().to_numpy() & ~d["obs_interpolada"].to_numpy()
        ERR[h[real], s[real], t[real]] = d["error_c"].to_numpy()[real]
    return estaciones, t0, ERR, TEMP, ROCP


def serie_obs(estaciones, t0, nT, archivo, col, filtro=None):
    """O[s, t]: valor observado REAL por estación y hora (NaN si no hay)."""
    o = pd.read_parquet(RAIZ / "data" / archivo)
    if filtro is not None:
        o = o[filtro(o)]
    s = estaciones.get_indexer(o["station_id"].astype(str))
    t = ((o["target_time_utc"] - t0) / pd.Timedelta(hours=1)).astype(int).to_numpy()
    ok = (s >= 0) & (t >= 0) & (t < nT)
    O = np.full((len(estaciones), nT), np.nan, dtype=np.float32)
    O[s[ok], t[ok]] = o[col].to_numpy()[ok]
    return O


def desvios_moviles(ERR, ventana, min_periodos):
    """Desvío (ddof=0) de ERR[h, s, t-ventana+1 .. t] con al menos min_periodos datos."""
    m1 = medias_moviles(ERR, ventana, min_periodos).astype(np.float64)
    m2 = medias_moviles(ERR.astype(np.float64) ** 2, ventana, min_periodos).astype(np.float64)
    return np.sqrt(np.clip(m2 - m1 ** 2, 0, None)).astype(np.float32)


def medias_moviles(ERR, ventana, min_periodos):
    """roll[h, s, t] = media de ERR[h, s, t-ventana+1 .. t] con al menos min_periodos datos."""
    ok = ~np.isnan(ERR)
    cs = np.cumsum(np.where(ok, ERR, 0.0), axis=2, dtype=np.float64)
    cn = np.cumsum(ok, axis=2, dtype=np.float64)
    cs_prev = np.concatenate([np.zeros_like(cs[:, :, :ventana]), cs[:, :, :-ventana]], axis=2)
    cn_prev = np.concatenate([np.zeros_like(cn[:, :, :ventana]), cn[:, :, :-ventana]], axis=2)
    n = cn - cn_prev
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (cs - cs_prev) / n
    r[n < min_periodos] = np.nan
    return r.astype(np.float32)


def tensor_ewma(ERR, t0, alfas):
    """S[alfa][estación, hora UTC 0-23, día]: EWMA de la media diaria del error (todos los horizontes)."""
    nS, nT = ERR.shape[1], ERR.shape[2]
    nD = nT // 24 + (1 if nT % 24 else 0)
    pad = nD * 24 - nT
    E = np.concatenate([ERR, np.full((NH, nS, pad), np.nan, dtype=np.float32)], axis=2) if pad else ERR
    E = E.reshape(NH, nS, nD, 24)                      # [h, s, día, hora]
    with warnings.catch_warnings():                    # horas sin ningún error real: NaN esperado
        warnings.simplefilter("ignore", RuntimeWarning)
        M = np.nanmean(E, axis=0)                      # [s, día, hora] media sobre horizontes
    M = np.transpose(M, (0, 2, 1))                     # [s, hora, día]
    out = {}
    for a in alfas:
        S = np.full_like(M, np.nan)
        for D in range(nD):
            previo = S[:, :, D - 1] if D > 0 else np.full(M.shape[:2], np.nan, dtype=M.dtype)
            m = M[:, :, D]
            nuevo = np.where(np.isnan(previo), m, np.where(np.isnan(m), previo, (1 - a) * previo + a * m))
            S[:, :, D] = nuevo
        out[a] = S
    return out


def elevacion_solar(t_utc, lat, lon):
    """Elevación del sol (grados) con las fórmulas aproximadas de la NOAA (error < 1° para este uso).
    No necesita datos externos: fecha y hora UTC, latitud y longitud (oeste negativa)."""
    t = pd.DatetimeIndex(t_utc)
    doy = t.day_of_year.to_numpy()
    hora = t.hour.to_numpy() + t.minute.to_numpy() / 60
    g = 2 * np.pi / 365 * (doy - 1 + (hora - 12) / 24)
    eq = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                   - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))          # ecuación del tiempo (min)
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))  # declinación (rad)
    tst = hora * 60 + eq + 4 * lon                                               # tiempo solar verdadero (min)
    ha = np.radians(tst / 4 - 180)                                               # ángulo horario
    la = np.radians(lat)
    cz = np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)
    return 90 - np.degrees(np.arccos(np.clip(cz, -1, 1)))


def tensor_media_pron(TEMP, dias=DIAS_ANOM, min_dias=MIN_DIAS_ANOM):
    """MP[h, s, t] = media de la temperatura pronosticada con el MISMO horizonte para la misma estación y hora,
    en los `dias` días anteriores (t-24, ..., t-24*dias), con al menos min_dias valores. Esos pronósticos se
    emitieron en t-24k-h, antes que el de la fila, así que se conocen al emitir."""
    nT = TEMP.shape[2]
    suma = np.zeros(TEMP.shape, dtype=np.float64)
    n = np.zeros(TEMP.shape, dtype=np.int16)
    for k in range(1, dias + 1):
        sh = np.full(TEMP.shape, np.nan, dtype=np.float32)
        sh[:, :, 24 * k:] = TEMP[:, :, :nT - 24 * k]
        ok = ~np.isnan(sh)
        suma += np.where(ok, sh, 0)
        n += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        mp = suma / n
    mp[n < min_dias] = np.nan
    return mp.astype(np.float32)


def vecinos(estaciones, k=VEC["k"], max_km=VEC["max_km"]):
    """Para cada estación, índices de sus k vecinas más cercanas a menos de max_km (-1 = no hay)."""
    c = pd.concat([pd.read_parquet(f, columns=["station_id", "latitude", "longitude"]).drop_duplicates("station_id")
                   for f in sorted(glob.glob(str(RAIZ / "data/pares/*.parquet")))])
    c = c.assign(station_id=c["station_id"].astype(str)).drop_duplicates("station_id").set_index("station_id")
    c = c.reindex(estaciones)
    la, lo = np.radians(c["latitude"].to_numpy()), np.radians(c["longitude"].to_numpy())
    a = (np.sin((la[:, None] - la[None]) / 2) ** 2
         + np.cos(la[:, None]) * np.cos(la[None]) * np.sin((lo[:, None] - lo[None]) / 2) ** 2)
    dist = 2 * 6371 * np.arcsin(np.sqrt(a))
    np.fill_diagonal(dist, np.inf)
    dist[np.isnan(dist)] = np.inf
    idx = np.argsort(dist, axis=1)[:, :k]
    idx[np.take_along_axis(dist, idx, axis=1) > max_km] = -1
    return idx


def tensor_vecinos(ERR, V):
    """NV[h, s, t] = media de ERR[h, v, t] sobre las vecinas v de s que tienen dato (NaN si ninguna)."""
    suma = np.zeros(ERR.shape, dtype=np.float32)
    n = np.zeros(ERR.shape, dtype=np.int8)
    for j in range(V.shape[1]):
        v = V[:, j]
        x = np.full(ERR.shape, np.nan, dtype=np.float32)
        x[:, v >= 0] = ERR[:, v[v >= 0]]
        ok = ~np.isnan(x)
        suma += np.where(ok, x, 0)
        n += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, suma / n, np.nan).astype(np.float32)


def features_dia(d, estaciones, t0, ERR, TEMP, R3, R6, EW, MP, NV, E1, OBS, ROC, ROCP, SD):
    s = estaciones.get_indexer(d["station_id"].astype(str))
    t = ((d["target_time_utc"] - t0) / pd.Timedelta(hours=1)).astype(int).to_numpy()
    h = d["horizon_target_hours"].to_numpy().astype(int)
    nT = ERR.shape[2]
    X = pd.DataFrame({"station_id": d["station_id"].astype(str).to_numpy(),
                      "target_time_utc": d["target_time_utc"].to_numpy(),
                      "forecast_generated_at_utc": d["forecast_generated_at_utc"].to_numpy(),
                      "horizonte": h.astype("int8"),
                      "error_c": d["error_c"].to_numpy(),
                      "obs_interpolada": d["obs_interpolada"].to_numpy()})
    # ---- 1. base
    f = d["temperature_forecast_c"].to_numpy()
    X["temp_pron"] = f
    X["hum_pron"] = d["relative_humidity_forecast_pct"].to_numpy()
    X["rocio_pron"] = d["dewpoint_forecast_c"].to_numpy()
    X["dep_rocio"] = f - d["dewpoint_forecast_c"].to_numpy()
    X["viento_vel"] = d["wind_speed_forecast_kmh"].to_numpy()
    ang = np.deg2rad(d["wind_direction_forecast"].astype(str).map(RUMBOS).astype(float).to_numpy())
    X["viento_calmo"] = np.isnan(ang)                      # sin dirección informada
    X["viento_sin"] = np.nan_to_num(np.sin(ang))
    X["viento_cos"] = np.nan_to_num(np.cos(ang))
    X["prob_lluvia"] = d["precip_probability_forecast_pct"].to_numpy()
    X["condicion"] = d["short_forecast"].astype(str).map(agrupar_condicion).astype("category")
    lon = d["longitude"].to_numpy()
    hs = (d["target_time_utc"].dt.hour.to_numpy() + d["target_time_utc"].dt.minute.to_numpy() / 60 + lon / 15) % 24
    X["hora_solar_sin"] = np.sin(2 * np.pi * hs / 24)
    X["hora_solar_cos"] = np.cos(2 * np.pi * hs / 24)
    X["lat"] = d["latitude"].to_numpy()
    X["lon"] = lon
    elev = elevacion_solar(d["target_time_utc"], d["latitude"].to_numpy(), lon)
    X["elev_solar"] = elev.astype(np.float32)
    X["noche"] = elev < SOL_HORIZONTE                   # sol bajo el horizonte (con refracción)
    # ---- 2. pendiente del pronóstico (mismo ciclo de emisión: (T+1, h+1) y (T-1, h-1))
    def tomar(T, hh, tt):
        ok = (hh >= 1) & (hh <= NH) & (tt >= 0) & (tt < nT)
        v = np.full(len(hh), np.nan, dtype=np.float32)
        v[ok] = T[hh[ok] - 1, s[ok], tt[ok]]
        return v
    a, b = tomar(TEMP, h + 1, t + 1), tomar(TEMP, h - 1, t - 1)
    with np.errstate(invalid="ignore"):
        X["pendiente_pron"] = np.where(~np.isnan(a) & ~np.isnan(b), (a - b) / 2,
                                       np.where(~np.isnan(a), a - f, np.where(~np.isnan(b), f - b, np.nan)))
    # anomalía: pronóstico menos la media de los pronósticos de la última semana (misma estación, hora y horizonte)
    X["temp_anom"] = f - tomar(MP, h, t)
    # ---- 3. rezagos (fuente más reciente: T - h - RETRASO; mismo horizonte)
    src = t - h - RETRASO
    X["err_ult"] = tomar(ERR, h, src)
    X["err_media3"] = tomar(R3, h, src)
    X["err_media6"] = tomar(R6, h, src)
    X["err_ayer"] = tomar(ERR, h, t - 24 * k_dias(h))
    # ---- 4. sesgo exponencial por estación y hora UTC de la celda, hasta el día  D_T - k
    D_T, u = t // 24, t % 24
    Dsrc = D_T - k_dias(h)
    for a_ in ALFAS:
        S = EW[a_]
        v = np.full(len(h), np.nan, dtype=np.float32)
        ok = Dsrc >= 0
        v[ok] = S[s[ok], u[ok], Dsrc[ok]]
        X[f"sesgo_ewma_{int(round(a_ * 100)):02d}"] = v
    # ---- 5. vecinos (misma fuente que los rezagos: T - h - RETRASO)
    X["vec_err_ult"] = tomar(NV, h, src)                       # mismo horizonte que la fila
    X["vec_err_1h"] = tomar(NV, np.ones_like(h), src)          # pronóstico a 1 h: lo más reciente de la zona
    # ---- 6. revisión: el pronóstico para la misma hora T emitido k horas antes tiene horizonte h+k
    #      (se emitió antes que el de la fila, así que se conocía). NaN si h+k > 48.
    for k in REVISION_H:
        X[f"rev_{k}h"] = f - tomar(TEMP, h + k, t)
    # ---- 7a. rezagos largos: mismo horizonte, k horas antes de la emisión (k > RETRASO, así que ya se conocían)
    for k in REZAGOS_LARGOS_H:
        X[f"err_hace{k}h"] = tomar(ERR, h, t - h - k)
    # ---- 7b. error del pronóstico a 1 h de la propia estación (E1[0] = último; E1[j] = media de VENTANAS_1H[j-1])
    uno = np.ones_like(h)
    X["err1h_ult"] = tomar(ERR, uno, src)
    for (v, _), M in zip(VENTANAS_1H, E1):
        X[f"err1h_media{v}"] = tomar(M, uno, src)
    # ---- 8. dinámica
    def tomar_obs(O, tt):
        ok = (tt >= 0) & (tt < nT)
        v = np.full(len(tt), np.nan, dtype=np.float32)
        v[ok] = O[s[ok], tt[ok]]
        return v
    o_src = tomar_obs(OBS, src)
    X["obs_tend3h"] = o_src - tomar_obs(OBS, src - 3)                        # a) tendencia
    X["err1h_tend3h"] = X["err1h_ult"].to_numpy() - tomar(ERR, uno, src - 3)
    X["salto_obs"] = f - o_src
    r_src = tomar_obs(ROC, src)                                              # b) rocío
    X["err_rocio_ult"] = r_src - tomar(ROCP, h, src)
    X["err_rocio_1h"] = r_src - tomar(ROCP, uno, src)
    for (v, _), M in zip(VENTANAS_STD, SD):                                  # c) desvío del error
        X[f"err_std{v}"] = tomar(M, h, src)
    with warnings.catch_warnings():                                          # nanstd/nanmax de filas vacías
        warnings.simplefilter("ignore", RuntimeWarning)
        D = np.column_stack([tomar(TEMP, h + j, t) for j in range(DISP_H + 1)])          # d) misma hora T
        X["dispersion_emis"] = np.where((~np.isnan(D)).sum(1) >= MIN_DISP, np.nanstd(D, axis=1), np.nan)
        A = np.column_stack([tomar(TEMP, h + j, t + j) for j in range(-AMP_H, AMP_H + 1)])  # e) misma emisión
        X["amplitud_pron"] = np.where((~np.isnan(A)).sum(1) >= MIN_AMP, np.nanmax(A, 1) - np.nanmin(A, 1), np.nan)
    return X


FAMILIAS = {
    "base": ["temp_pron", "temp_anom", "elev_solar", "noche", "hum_pron", "rocio_pron", "dep_rocio", "viento_vel",
             "viento_sin", "viento_cos",
             "viento_calmo", "prob_lluvia", "condicion", "horizonte", "hora_solar_sin", "hora_solar_cos", "lat", "lon"],
    "pendiente": ["pendiente_pron"],
    "rezagos": ["err_ult", "err_media3", "err_media6", "err_ayer"],
    "sesgo_ewma": [f"sesgo_ewma_{int(round(a * 100)):02d}" for a in ALFAS],
    "vecinos": ["vec_err_ult", "vec_err_1h"],
    "revision": [f"rev_{k}h" for k in REVISION_H],
    "rezagos_extra": [f"err_hace{k}h" for k in REZAGOS_LARGOS_H] + ["err1h_ult"] + [f"err1h_media{v}" for v, _ in VENTANAS_1H],
    "dinamica": ["obs_tend3h", "err1h_tend3h", "salto_obs", "err_rocio_ult", "err_rocio_1h"]
                + [f"err_std{v}" for v, _ in VENTANAS_STD] + ["dispersion_emis", "amplitud_pron"],
}


def main():
    salida = RAIZ / "data" / "features"
    salida.mkdir(exist_ok=True)
    print("Calcula las variables que usan los modelos, solo con lo que se sabía al emitir cada pronóstico (nada del\n"
          "futuro). Familias que usan los modelos del producto:\n"
          "  base        el pronóstico mismo (temperatura, humedad, rocío, viento, lluvia, condición), horizonte,\n"
          "              hora del día, ubicación, altura del sol y si es de noche\n"
          "  pendiente   cuánto cambia la temperatura pronosticada alrededor de la hora objetivo\n"
          "  rezagos     últimos errores conocidos de la misma estación y el mismo horizonte\n"
          "  sesgo EWMA  sesgo habitual de la estación a esa hora, promediando días con más peso a los recientes\n"
          "Familias experimentales (se calculan, pero solo se usan al probarlas): vecinos, revisión, rezagos extra,\n"
          "dinámica.")
    estaciones, t0, ERR, TEMP, ROCP = construir_tensores()
    nT = ERR.shape[2]
    OBS = serie_obs(estaciones, t0, nT, "obs_horaria.parquet", "temp_obs_c", lambda o: ~o["obs_interpolada"])
    ROC = serie_obs(estaciones, t0, nT, "obs_rocio.parquet", "dewpoint_observed_c")
    SD = [desvios_moviles(ERR, v, m) for v, m in VENTANAS_STD]
    R3, R6 = medias_moviles(ERR, 3, 2), medias_moviles(ERR, 6, 3)
    EW = tensor_ewma(ERR, t0, ALFAS)
    MP = tensor_media_pron(TEMP)
    V = vecinos(estaciones)
    print(f"\nEstaciones vecinas (las {VEC['k']} más cercanas a menos de {VEC['max_km']} km, para la familia «vecinos»): "
          f"{dec((V >= 0).sum(1).mean())} por estación\nen promedio; {int((V[:, 0] < 0).sum())} de {len(estaciones)} "
          "estaciones no tienen ninguna a menos de esa distancia.")
    NV = tensor_vecinos(ERR, V)
    E1 = [medias_moviles(ERR[:1], v, m) for v, m in VENTANAS_1H]      # solo el horizonte 1 h
    for p in sorted((RAIZ / "data" / "pares").glob("*.parquet")):
        d = pd.read_parquet(p, columns=COLS_LECTURA)
        features_dia(d, estaciones, t0, ERR, TEMP, R3, R6, EW, MP, NV, E1, OBS, ROC, ROCP, SD).to_parquet(salida / p.name, index=False)
    (salida / "familias.json").write_text(json.dumps(FAMILIAS, indent=1))
    print(f"Variables calculadas para {len(list(salida.glob('*.parquet')))} días → data/features/")


if __name__ == "__main__":
    main()
