"""Feature engineering del TP3 (familias 1 a 4 acordadas el 2/11/2026).

Todas las variables se calculan con lo que existía AL EMITIR el pronóstico; no hay nada
que "aprender" del entrenamiento, así que se generan una sola vez para todas las filas
(data/features/<dia>.parquet) y los pliegues solo eligen qué días usar.

  1. Base         pronóstico y sus variables, horizonte, hora solar (sen/cos), ubicación, elevación del
                  sol y noche (sol bajo el horizonte), anomalía de la temperatura pronosticada
  2. Pendiente    cuánto cambia la temperatura pronosticada alrededor de la hora objetivo
  3. Rezagos      últimos errores conocidos de la misma estación y el mismo horizonte
  4. Sesgo EWMA   promedio exponencial de días del error por estación y hora (tipo NBM)

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

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((RAIZ / "config.yaml").read_text())
ALFAS = CFG["features"]["alfas_ewma"]
RETRASO = 2                                  # horas de seguridad (ver REGLA DE CAUSALIDAD)
NH = 48                                      # horizontes 1..48
DIAS_ANOM, MIN_DIAS_ANOM = 7, 3              # anomalía de temperatura: media de los pronósticos de 7 días previos
SOL_HORIZONTE = -0.833                       # elevación del sol al salir/ponerse (refracción + radio del disco)

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
    for f in archivos:
        d = pd.read_parquet(f, columns=["station_id", "target_time_utc", "horizon_target_hours",
                                        "temperature_forecast_c", "error_c", "obs_interpolada"])
        s = estaciones.get_indexer(d["station_id"].astype(str))
        t = ((d["target_time_utc"] - t0) / pd.Timedelta(hours=1)).astype(int).to_numpy()
        h = d["horizon_target_hours"].to_numpy().astype(int) - 1
        TEMP[h, s, t] = d["temperature_forecast_c"].to_numpy()
        real = d["error_c"].notna().to_numpy() & ~d["obs_interpolada"].to_numpy()
        ERR[h[real], s[real], t[real]] = d["error_c"].to_numpy()[real]
    return estaciones, t0, ERR, TEMP


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
    with np.errstate(invalid="ignore"):
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


def features_dia(d, estaciones, t0, ERR, TEMP, R3, R6, EW, MP):
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
    return X


FAMILIAS = {
    "base": ["temp_pron", "temp_anom", "elev_solar", "noche", "hum_pron", "rocio_pron", "dep_rocio", "viento_vel",
             "viento_sin", "viento_cos",
             "viento_calmo", "prob_lluvia", "condicion", "horizonte", "hora_solar_sin", "hora_solar_cos", "lat", "lon"],
    "pendiente": ["pendiente_pron"],
    "rezagos": ["err_ult", "err_media3", "err_media6", "err_ayer"],
    "sesgo_ewma": [f"sesgo_ewma_{int(round(a * 100)):02d}" for a in ALFAS],
}


def main():
    salida = RAIZ / "data" / "features"
    salida.mkdir(exist_ok=True)
    print("construyendo tensores...")
    estaciones, t0, ERR, TEMP = construir_tensores()
    R3, R6 = medias_moviles(ERR, 3, 2), medias_moviles(ERR, 6, 3)
    EW = tensor_ewma(ERR, t0, ALFAS)
    MP = tensor_media_pron(TEMP)
    for p in sorted((RAIZ / "data" / "pares").glob("*.parquet")):
        d = pd.read_parquet(p, columns=COLS_LECTURA)
        features_dia(d, estaciones, t0, ERR, TEMP, R3, R6, EW, MP).to_parquet(salida / p.name, index=False)
    (salida / "familias.json").write_text(json.dumps(FAMILIAS, indent=1))
    print("listo:", len(list(salida.glob('*.parquet'))), "días")


if __name__ == "__main__":
    main()
