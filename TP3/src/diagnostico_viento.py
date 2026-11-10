"""Diagnóstico: ¿la dirección del viento (pronosticada) explica error que el sesgo por estación × hora no capta?

Hipótesis física: el viento trae las propiedades del lugar de donde viene (mar: húmedo y templado; montaña:
seco y, al bajar, más cálido; aire frío atrapado al este de los Apalaches con viento del NE). El efecto
depende de la geografía de cada estación, así que se mide por ESTACIÓN × SECTOR, no con un coeficiente global.

Método (solo en los pliegues de ELECCIÓN; los de reporte no se tocan):
  - Para cada pliegue de elección: con su entrenamiento se aprende B2 (estación × hora solar) y, sobre el
    residuo, un efecto por estación × sector de viento (8 sectores + calma), CONTRAÍDO hacia 0:
        efecto = suma(residuo) / (n + k)
    k regula cuánto hay que ver antes de creerle a una celda. Las filas se repiten por horizonte, así que
    n está inflado: k se expresa en filas y se prueban varios valores.
  - Se mide en la prueba del pliegue si B2 + efecto de sector baja el MAE frente a B2 solo.
  - Como control, el mismo cálculo con un efecto GLOBAL por sector (sin estación): se espera ~0.

Uso:  .venv/bin/python src/diagnostico_viento.py
"""
import numpy as np
import pandas as pd

from modelado import RAIZ, hora_solar, pliegues_eleccion_reporte, _archivo

SECTORES = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
KS = [0, 200, 1000, 5000, 20000]
COLS = ["station_id", "error_c", "obs_interpolada", "hora_solar_sin", "hora_solar_cos",
        "viento_sin", "viento_cos", "viento_calmo", "horizonte"]


def cargar(dias):
    partes = []
    for d in dias:
        x = pd.read_parquet(_archivo(d), columns=COLS)
        x = x[x["error_c"].notna() & ~x["obs_interpolada"]]
        ang = np.degrees(np.arctan2(x["viento_sin"].to_numpy(), x["viento_cos"].to_numpy())) % 360
        sector = ((ang + 22.5) // 45).astype(int) % 8
        sector = np.where(x["viento_calmo"].to_numpy(), 8, sector)          # 8 = calma
        partes.append(pd.DataFrame({"est": x["station_id"].astype(str).to_numpy(), "hora": hora_solar(x),
                                    "sector": sector.astype("int8"), "h": x["horizonte"].to_numpy(),
                                    "e": x["error_c"].to_numpy(dtype=np.float32)}))
    d = pd.concat(partes, ignore_index=True)
    d["est"] = d["est"].astype("category")
    return d


def main():
    eleccion, _ = pliegues_eleccion_reporte()
    filas, amplitudes = [], []
    for k_pl, (tr, te) in eleccion:
        A, T = cargar(tr), cargar(te)
        T["est"] = T["est"].astype(str)
        A_est = A["est"].astype(str)
        b2 = A.assign(est=A_est).groupby(["est", "hora"])["e"].mean().rename("b2")
        A = A.assign(est=A_est).join(b2, on=["est", "hora"])
        T = T.join(b2, on=["est", "hora"])
        T = T[T["b2"].notna()]
        A["res"] = A["e"] - A["b2"]
        base = np.abs(T["e"] - T["b2"]).mean()
        filas.append({"pliegue": k_pl, "variante": "B2 (estación × hora)", "k": np.nan, "MAE": base})

        # control: efecto global por sector
        g = A.groupby("sector")["res"].mean()
        r = T["e"] - T["b2"] - T["sector"].map(g).fillna(0)
        filas.append({"pliegue": k_pl, "variante": "B2 + sector global", "k": np.nan, "MAE": np.abs(r).mean()})
        if k_pl == eleccion[0][0]:
            print("efecto global por sector (residuo medio, °C):",
                  {(SECTORES + ["calma"])[s]: round(v, 3) for s, v in g.items()})

        # efecto por estación × sector, contraído
        agg = A.groupby(["est", "sector"])["res"].agg(["sum", "size"])
        for k in KS:
            ef = (agg["sum"] / (agg["size"] + k)).rename("ef")
            c = T.join(ef, on=["est", "sector"])["ef"].fillna(0)
            filas.append({"pliegue": k_pl, "variante": "B2 + estación × sector", "k": k,
                          "MAE": np.abs(T["e"] - T["b2"] - c).mean()})
        # amplitud por estación (con k intermedio): diferencia entre su peor y su mejor sector
        ef = agg["sum"] / (agg["size"] + 1000)
        ef = ef[agg["size"] >= 2000]
        amp = ef.groupby(level="est", observed=True).agg(lambda s: s.max() - s.min())
        amplitudes.append(amp.rename(f"pliegue_{k_pl}"))
        print(f"pliegue {k_pl}: entrenamiento {len(A):,} | prueba {len(T):,} filas reales", flush=True)

    R = pd.DataFrame(filas)
    res = R.groupby(["variante", "k"], dropna=False)["MAE"].mean().reset_index()
    base = res.loc[res.variante.str.startswith("B2 (")]["MAE"].iloc[0]
    res["mejora_vs_B2_%"] = 100 * (base - res["MAE"]) / base
    print("\nMAE medio en la prueba de los pliegues de elección (°C):")
    print(res.round(4).to_string(index=False))

    amp = pd.concat(amplitudes, axis=1).mean(axis=1).sort_values(ascending=False)
    print(f"\namplitud entre sectores por estación (k=1000, celdas ≥ 2000 filas): mediana {amp.median():.2f} °C, "
          f"p90 {amp.quantile(.9):.2f} °C, {len(amp)} estaciones")
    est = pd.read_parquet(_archivo(eleccion[0][1][0][0]), columns=["station_id", "lat", "lon"]).drop_duplicates("station_id")
    top = amp.head(12).rename("amplitud_C").reset_index().rename(columns={"index": "est"})
    top = top.merge(est.rename(columns={"station_id": "est"}).assign(est=lambda x: x.est.astype(str)), on="est", how="left")
    print("\nestaciones con más diferencia entre sectores:")
    print(top.round(2).to_string(index=False))
    salida = RAIZ / "data" / "resultados"
    res.to_csv(salida / "diagnostico_viento.csv", index=False)
    amp.rename("amplitud_C").to_csv(salida / "diagnostico_viento_amplitud.csv")


if __name__ == "__main__":
    main()
