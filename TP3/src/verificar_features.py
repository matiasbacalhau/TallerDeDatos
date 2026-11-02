"""Control independiente de features.py: recalcula a mano (con pandas, sin tensores) una muestra
de filas y comprueba (1) que los valores coinciden y (2) que ninguna fuente es posterior a la emisión.

Uso:  .venv/bin/python src/verificar_features.py
"""
from pathlib import Path
import glob

import numpy as np
import pandas as pd

from features import ALFAS, RETRASO, k_dias, FAMILIAS

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
    print(f"causalidad: {viola} filas de {len(F):,} usan una fuente posterior a la emisión")
    assert viola == 0

    # (2) recálculo independiente en una muestra de estaciones
    est = rng.choice(F["station_id"].unique(), n_estaciones, replace=False)
    P = pd.concat([pd.read_parquet(p, columns=["station_id", "target_time_utc", "horizon_target_hours",
                                                "temperature_forecast_c", "error_c", "obs_interpolada"])
                   for p in sorted((RAIZ / "data/pares").glob("*.parquet"))], ignore_index=True)
    P["station_id"] = P["station_id"].astype(str)
    P = P[P["station_id"].isin(est)]
    real = P[P["error_c"].notna() & ~P["obs_interpolada"]]
    err = real.set_index(["station_id", "target_time_utc", "horizon_target_hours"])["error_c"]
    tmp = P.set_index(["station_id", "target_time_utc", "horizon_target_hours"])["temperature_forecast_c"]
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
    errores = {c: 0 for c in ["err_ult", "err_media3", "err_media6", "err_ayer", "pendiente_pron"] + FAMILIAS["sesgo_ewma"]}
    for _, r in M.iterrows():
        s, T, h = r["station_id"], r["target_time_utc"], int(r["h"])
        fuente = T - (h + RETRASO) * H
        esperados = {"err_ult": e(s, fuente, h),
                     "err_media3": np.nanmean([e(s, fuente - j * H, h) for j in range(3)]) if sum(~np.isnan([e(s, fuente - j * H, h) for j in range(3)])) >= 2 else np.nan,
                     "err_media6": np.nanmean([e(s, fuente - j * H, h) for j in range(6)]) if sum(~np.isnan([e(s, fuente - j * H, h) for j in range(6)])) >= 3 else np.nan,
                     "err_ayer": e(s, T - 24 * int(k_dias(np.array([h]))[0]) * H, h)}
        f = r["temp_pron"]
        a_ = tmp.get((s, T + H, h + 1), np.nan)
        b_ = tmp.get((s, T - H, h - 1), np.nan)
        esperados["pendiente_pron"] = ((a_ - b_) / 2 if not (np.isnan(a_) or np.isnan(b_))
                                       else (a_ - f if not np.isnan(a_) else (f - b_ if not np.isnan(b_) else np.nan)))
        dmax = (T - int(k_dias(np.array([h]))[0]) * pd.Timedelta(days=1)).normalize()
        for al in ALFAS:
            esperados[f"sesgo_ewma_{int(round(al * 100)):02d}"] = ewma(s, T.hour, dmax, al)
        for c, v in esperados.items():
            x = r[c]
            if not ((np.isnan(v) and np.isnan(x)) or abs(v - x) < 1e-3):
                errores[c] += 1
    print(f"recálculo independiente de {n_filas} filas ({n_estaciones} estaciones):")
    for c, n in errores.items():
        print(f"  {c:<16} discrepancias: {n}")
    assert sum(errores.values()) == 0
    print("\nfaltantes por variable (% de filas):")
    print((F[sum(FAMILIAS.values(), [])].isna().mean() * 100).round(1).to_string())
    print("\ncondición:", F["condicion"].value_counts(normalize=True).round(3).to_dict())
    print("viento calmo (sin dirección):", round(F["viento_calmo"].mean(), 3),
          "| velocidad media con dirección faltante:", round(F.loc[F["viento_calmo"], "viento_vel"].mean(), 2),
          "km/h, con dirección:", round(F.loc[~F["viento_calmo"], "viento_vel"].mean(), 2))


if __name__ == "__main__":
    main()
