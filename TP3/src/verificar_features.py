"""Control independiente de features.py: recalcula a mano (con pandas, sin tensores) una muestra
de filas y comprueba (1) que los valores coinciden y (2) que ninguna fuente es posterior a la emisión.

Uso:  .venv/bin/python src/verificar_features.py
"""
from pathlib import Path
import glob
import textwrap

import numpy as np
import pandas as pd

from sklearn.neighbors import BallTree

from formato import n, dec, pct

from features import ALFAS, RETRASO, k_dias, FAMILIAS, DIAS_ANOM, MIN_DIAS_ANOM, VEC, REVISION_H, REZAGOS_LARGOS_H, VENTANAS_1H, \
    VENTANAS_STD, DISP_H, MIN_DISP, AMP_H, MIN_AMP

RAIZ = Path(__file__).resolve().parents[1]
H = pd.Timedelta(hours=1)


def main(n_estaciones=25, n_filas=1500, semilla=42):
    rng = np.random.default_rng(semilla)
    F = pd.concat([pd.read_parquet(p) for p in sorted((RAIZ / "data/features").glob("*.parquet"))], ignore_index=True)
    F["h"] = F["horizonte"].astype(int)

    # (1) causalidad sobre TODAS las filas: la fuente más reciente es T - h - RETRASO horas y puede
    # llegar hasta 59 min después de su hora; tiene que estar antes de la emisión.
    ultima_fuente = F["target_time_utc"] - (F["h"] + RETRASO) * H + pd.Timedelta(minutes=59)
    viola = int((ultima_fuente > F["forecast_generated_at_utc"]).sum())
    print("Revisa las variables del paso anterior por otro camino: si el cálculo original tuviera un error, acá no\n"
          "coincidiría. Si algún control falla, el pipeline se detiene.\n")
    print(f"1. Causalidad: de {n(len(F))} filas, {n(viola)} usan un dato observado después de emitido el pronóstico → "
          + ("OK" if viola == 0 else "FALLA"))
    assert viola == 0

    # (2) recálculo independiente en una muestra de estaciones
    est = rng.choice(F["station_id"].unique(), n_estaciones, replace=False)
    P = pd.concat([pd.read_parquet(p, columns=["station_id", "target_time_utc", "horizon_target_hours",
                                                "temperature_forecast_c", "dewpoint_forecast_c", "error_c", "obs_interpolada",
                                                "forecast_generated_at_utc", "latitude", "longitude"])
                   for p in sorted((RAIZ / "data/pares").glob("*.parquet"))], ignore_index=True)
    P["station_id"] = P["station_id"].astype(str)
    # vecinas con otro método (BallTree con distancia haversine) para el control de la familia 5
    coord = P.drop_duplicates("station_id").set_index("station_id")[["latitude", "longitude"]]
    arbol = BallTree(np.radians(coord.to_numpy()), metric="haversine")
    dist, ind = arbol.query(np.radians(coord.loc[est].to_numpy()), k=VEC["k"] + 1)
    vecinas = {s: [coord.index[i] for d, i in zip(dist[j][1:], ind[j][1:]) if d * 6371 <= VEC["max_km"]]
               for j, s in enumerate(est)}
    P = P[P["station_id"].isin(set(est) | {v for vs in vecinas.values() for v in vs})]
    real = P[P["error_c"].notna() & ~P["obs_interpolada"]]
    err = real.set_index(["station_id", "target_time_utc", "horizon_target_hours"])["error_c"]
    tmp = P.set_index(["station_id", "target_time_utc", "horizon_target_hours"])["temperature_forecast_c"]
    gen = P.set_index(["station_id", "target_time_utc", "horizon_target_hours"])["forecast_generated_at_utc"]
    gen = gen[~gen.index.duplicated()]
    tmp = tmp[~tmp.index.duplicated()]
    rocp = P.set_index(["station_id", "target_time_utc", "horizon_target_hours"])["dewpoint_forecast_c"]
    rocp = rocp[~rocp.index.duplicated()]
    oh = pd.read_parquet(RAIZ / "data/obs_horaria.parquet")
    obs = oh[~oh["obs_interpolada"]].assign(station_id=lambda x: x["station_id"].astype(str)) \
            .set_index(["station_id", "target_time_utc"])["temp_obs_c"]
    roc = pd.read_parquet(RAIZ / "data/obs_rocio.parquet").set_index(["station_id", "target_time_utc"])["dewpoint_observed_c"]
    diario = real.assign(dia=real["target_time_utc"].dt.normalize(), hora=real["target_time_utc"].dt.hour) \
                 .groupby(["station_id", "hora", "dia"])["error_c"].mean()

    def e(s, t, h):
        return err.get((s, t, h), np.nan)

    def ewma(s, u, dia_max, a):
        serie = diario.get((s, u), pd.Series(dtype=float)) if (s, u) in diario.index.droplevel(2) else pd.Series(dtype=float)
        v = np.nan
        for dia, m in serie.sort_index().items():
            if dia > dia_max:
                break
            v = m if np.isnan(v) else (1 - a) * v + a * m
        return v

    M = F[F["station_id"].isin(est)].sample(n_filas, random_state=semilla)
    errores = {c: 0 for c in ["err_ult", "err_media3", "err_media6", "err_ayer", "pendiente_pron", "temp_anom"]
               + FAMILIAS["sesgo_ewma"] + FAMILIAS["vecinos"] + FAMILIAS["revision"] + FAMILIAS["rezagos_extra"]
               + FAMILIAS["dinamica"]}
    anom_posterior = rev_posterior = 0
    for _, r in M.iterrows():
        s, T, h = r["station_id"], r["target_time_utc"], int(r["h"])
        fuente = T - (h + RETRASO) * H
        esperados = {"err_ult": e(s, fuente, h),
                     "err_media3": np.nanmean([e(s, fuente - j * H, h) for j in range(3)]) if sum(~np.isnan([e(s, fuente - j * H, h) for j in range(3)])) >= 2 else np.nan,
                     "err_media6": np.nanmean([e(s, fuente - j * H, h) for j in range(6)]) if sum(~np.isnan([e(s, fuente - j * H, h) for j in range(6)])) >= 3 else np.nan,
                     "err_ayer": e(s, T - 24 * int(k_dias(np.array([h]))[0]) * H, h)}
        for c, hv in (("vec_err_ult", h), ("vec_err_1h", 1)):
            vals = [e(v, fuente, hv) for v in vecinas[s]]
            esperados[c] = np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan
        f = r["temp_pron"]
        a_ = tmp.get((s, T + H, h + 1), np.nan)
        b_ = tmp.get((s, T - H, h - 1), np.nan)
        esperados["pendiente_pron"] = ((a_ - b_) / 2 if not (np.isnan(a_) or np.isnan(b_))
                                       else (a_ - f if not np.isnan(a_) else (f - b_ if not np.isnan(b_) else np.nan)))
        previos = [tmp.get((s, T - 24 * k * H, h), np.nan) for k in range(1, DIAS_ANOM + 1)]
        validos = [v for v in previos if not np.isnan(v)]
        esperados["temp_anom"] = f - np.mean(validos) if len(validos) >= MIN_DIAS_ANOM else np.nan
        # causalidad de la anomalía: los pronósticos promediados se emitieron antes que el de la fila
        for k in range(1, DIAS_ANOM + 1):
            g = gen.get((s, T - 24 * k * H, h), pd.NaT)
            if not pd.isna(g) and g > r["forecast_generated_at_utc"]:
                anom_posterior += 1
        # familia 7: rezagos largos (mismo horizonte) y error del pronóstico a 1 h de la propia estación
        for k in REZAGOS_LARGOS_H:
            esperados[f"err_hace{k}h"] = e(s, T - (h + k) * H, h)
        esperados["err1h_ult"] = e(s, fuente, 1)
        for v, m in VENTANAS_1H:
            vals = np.array([e(s, fuente - j * H, 1) for j in range(v)])
            esperados[f"err1h_media{v}"] = np.nanmean(vals) if (~np.isnan(vals)).sum() >= m else np.nan
        # familia 6: pronóstico para la misma hora T con horizonte h+k; tiene que haberse emitido antes
        for k in REVISION_H:
            esperados[f"rev_{k}h"] = f - tmp.get((s, T, h + k), np.nan)
            g = gen.get((s, T, h + k), pd.NaT)
            if not pd.isna(g) and g > r["forecast_generated_at_utc"]:
                rev_posterior += 1
        # familia 8: dinámica
        o = lambda t: obs.get((s, t), np.nan)
        esperados["obs_tend3h"] = o(fuente) - o(fuente - 3 * H)
        esperados["err1h_tend3h"] = e(s, fuente, 1) - e(s, fuente - 3 * H, 1)
        esperados["salto_obs"] = f - o(fuente)
        ro = roc.get((s, fuente), np.nan)
        esperados["err_rocio_ult"] = ro - rocp.get((s, fuente, h), np.nan)
        esperados["err_rocio_1h"] = ro - rocp.get((s, fuente, 1), np.nan)
        for v, m in VENTANAS_STD:
            vals = np.array([e(s, fuente - j * H, h) for j in range(v)])
            esperados[f"err_std{v}"] = np.nanstd(vals) if (~np.isnan(vals)).sum() >= m else np.nan
        vals = np.array([tmp.get((s, T, h + j), np.nan) if h + j <= 48 else np.nan for j in range(DISP_H + 1)])
        esperados["dispersion_emis"] = np.nanstd(vals) if (~np.isnan(vals)).sum() >= MIN_DISP else np.nan
        vals = np.array([tmp.get((s, T + j * H, h + j), np.nan) for j in range(-AMP_H, AMP_H + 1)])
        esperados["amplitud_pron"] = np.nanmax(vals) - np.nanmin(vals) if (~np.isnan(vals)).sum() >= MIN_AMP else np.nan
        dmax = (T - int(k_dias(np.array([h]))[0]) * pd.Timedelta(days=1)).normalize()
        for al in ALFAS:
            esperados[f"sesgo_ewma_{int(round(al * 100)):02d}"] = ewma(s, T.hour, dmax, al)
        for c, v in esperados.items():
            x = r[c]
            if not ((np.isnan(v) and np.isnan(x)) or abs(v - x) < 1e-3):
                errores[c] += 1
    malas = {c: k for c, k in errores.items() if k}
    print(f"\n2. Recálculo a mano (pandas fila por fila, vecinas con otro método) de {n(n_filas)} filas al azar de "
          f"{n_estaciones} estaciones,\n   {len(errores)} variables. Filas donde el valor original y el recalculado "
          "difieren en más de 0,001:")
    print("   0 filas en todas las variables → OK" if not malas else
          "\n".join(f"   {c}: {k} filas → FALLA" for c, k in malas.items()))
    print(f"   Variables que miran pronósticos anteriores (anomalía de temperatura, revisión): {anom_posterior + rev_posterior} "
          "usan un pronóstico emitido\n   después del de la fila → " + ("OK" if anom_posterior + rev_posterior == 0 else "FALLA"))
    assert sum(errores.values()) == 0 and anom_posterior == 0 and rev_posterior == 0

    # (3) noche: control físico de la elevación solar (independiente de la fórmula)
    hs = (np.degrees(np.arctan2(F["hora_solar_sin"], F["hora_solar_cos"])) / 15) % 24
    mediodia = F.loc[(hs > 11.5) & (hs < 12.5), "elev_solar"]
    medianoche = F.loc[(hs < 0.5) | (hs > 23.5), "elev_solar"]
    horas_noche = F.groupby([F["station_id"], F["target_time_utc"].dt.normalize()])["noche"].mean().mul(24).median()
    print("\n3. Altura del sol y noche (control físico, sin usar la fórmula del cálculo):")
    print(f"   al mediodía solar el sol tiene que estar sobre el horizonte: altura mínima {dec(mediodia.min())}° → " + ("OK" if mediodia.min() > 0 else "FALLA"))
    print(f"   a la medianoche solar tiene que estar debajo: altura máxima {dec(medianoche.max())}° → " + ("OK" if medianoche.max() < 0 else "FALLA"))
    print(f"   horas de noche por día (mediana entre estaciones y días): {dec(horas_noche)} h;\n"
          "   entre 10 y 14 h es lo esperable en estas latitudes y fechas → " + ("OK" if 10 < horas_noche < 14 else "FALLA"))
    assert mediodia.min() > 0 and medianoche.max() < 0 and 10 < horas_noche < 14

    print("\nInformativo (no detiene el pipeline):")
    print("\n4. Variables sin valor (% de filas). Es esperable: las que miran errores pasados quedan vacías si no hubo\n"
          "   observación real en ese momento, y las que necesitan varios días previos, al comienzo del período.")
    falt = F[sum(FAMILIAS.values(), [])].isna().mean()
    for fam, cols in FAMILIAS.items():
        con = [(c, falt[c]) for c in cols if falt[c] > 0]
        texto = "sin faltantes" if not con else ", ".join(f"{c}\u00a0{pct(v) if v >= 0.001 else '< 0,1 %'}".replace(" ", "\u00a0") for c, v in con)
        print(textwrap.fill(texto, 112, initial_indent=f"   {fam:<14}", subsequent_indent=" " * 17))

    print("\n5. Condición del tiempo que pronosticó el NWS (sus textos agrupados por palabras clave), % de filas:")
    for c, v in F["condicion"].value_counts(normalize=True).items():
        print(f"   {c:<10} {pct(v):>7}")

    calmo = F["viento_calmo"]
    print(f"\n6. Viento sin dirección en el pronóstico: {pct(calmo.mean())} de las filas. Velocidad media pronosticada: "
          f"{dec(F.loc[calmo, 'viento_vel'].mean())} km/h\n   en esas filas contra {dec(F.loc[~calmo, 'viento_vel'].mean())} km/h "
          "en las que tienen dirección → la dirección falta porque el viento es calmo,\n   no por un dato perdido.")


if __name__ == "__main__":
    main()
