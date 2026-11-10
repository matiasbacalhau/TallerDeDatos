"""Control de calidad por día: decide automáticamente qué días se excluyen del análisis.

Por qué existe: del 19 al 26/10/2026 el NWS dejó de entregar observaciones para una parte
grande de las estaciones (confirmado por el grupo; se acordó con el docente sacar esos días).
El producto recibe datos nuevos todo el tiempo, así que la decisión no puede ser una lista
escrita a mano: tiene que salir de una regla que se aplica a cada día que llega.

REGLA DE EXCLUSIÓN (relativa)
  Un día se excluye si su % de filas sin observación supera en más de `max_exceso_pp` puntos
  la mediana de los `ventana_dias` días VÁLIDOS anteriores. Se usa una referencia móvil y no un
  umbral fijo porque el nivel normal cambia: en septiembre ~25 % de filas sin observación, en
  octubre ~7 %. Solo cuentan días válidos en la referencia, así una falla larga no se vuelve
  «normal». Con menos de `min_dias_referencia` días previos no se excluye nada.

ALARMAS (solo avisan y quedan registradas; no excluyen)
  - por día: un día incluido con cobertura total (observación real + interpolada) menor a
    `alarma_cobertura_min`. Detecta degradaciones lentas que la regla relativa va absorbiendo.
  - del sistema: en los últimos 14 días de calendario, la fracción de días usables (con archivo
    y no excluidos) es menor a `alarma_dias_utiles_min`.

HISTORIA INCOMPLETA (marca, no excluye)
  Un día válido tiene historia incompleta si alguno de los `dias_historia` días de calendario
  anteriores falta (sin archivo) o fue excluido. Su target es bueno, pero sus features con memoria
  (rezagos) quedan vacías en buena parte. Se evalúa aparte: la comparación limpia entre modelos usa
  los días con historia completa; los de historia incompleta miden la robustez ante caídas.

Salidas (data/calidad/): calidad_por_dia.csv, dias_excluidos.json, historia_incompleta.json,
REGISTRO_CALIDAD.md.
Uso:  .venv/bin/python src/calidad_dias.py
"""
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((RAIZ / "config.yaml").read_text())
Q = CFG["calidad"]
SALIDA = RAIZ / CFG["salida"] / "calidad"
VENTANA_SISTEMA = 14          # días de calendario para la alarma del sistema


def medir_dia(ruta):
    """Métricas de faltantes de un día de data/pares/."""
    p = pd.read_parquet(ruta, columns=["station_id", "target_time_utc", "error_c", "obs_interpolada"])
    con = p["error_c"].notna()
    real = con & ~p["obs_interpolada"].astype(bool)
    st = p["station_id"].astype(str)
    estaciones = st.nunique()
    sin_obs_est = estaciones - st[con].nunique()
    # horas en las que más del 90 % de las estaciones no tiene dato (apagones de la fuente)
    por_hora = p.assign(con=con).groupby("target_time_utc")["con"].mean()
    return {"dia": ruta.stem, "filas": len(p),
            "pct_sin_obs": 100 * (1 - con.mean()),
            "cobertura_total": con.mean(), "cobertura_real": real.mean(),
            "estaciones": estaciones, "estaciones_sin_obs": sin_obs_est,
            "horas_apagon": int((por_hora < 0.10).sum())}


def aplicar_regla(m):
    """Recorre los días en orden y decide; la referencia usa solo días válidos anteriores."""
    validos, filas = [], []
    for r in m.to_dict("records"):
        previos = validos[-Q["ventana_dias"]:]
        if len(previos) < Q["min_dias_referencia"]:
            ref, umbral, excluir, motivo = np.nan, np.nan, False, "sin referencia (pocos días previos)"
        else:
            ref = float(np.median(previos))
            umbral = ref + Q["max_exceso_pp"]
            excluir = r["pct_sin_obs"] > umbral
            motivo = (f"{r['pct_sin_obs']:.1f} % sin observación > referencia {ref:.1f} % + "
                      f"{Q['max_exceso_pp']} pp" if excluir else "ok")
        if not excluir:
            validos.append(r["pct_sin_obs"])
        alarma = (not excluir) and r["cobertura_total"] < Q["alarma_cobertura_min"]
        filas.append({**r, "referencia_pct": ref, "umbral_pct": umbral, "excluido": excluir,
                      "motivo": motivo, "alarma_cobertura": alarma})
    return pd.DataFrame(filas)


def alarma_sistema(t):
    """Fracción de días usables en los últimos 14 días de calendario (hasta el último día con datos)."""
    fechas = pd.to_datetime(t["dia"], format="%Y_%m_%d")
    fin = fechas.max()
    ventana = pd.date_range(fin - pd.Timedelta(days=VENTANA_SISTEMA - 1), fin)
    usables = set(fechas[~t["excluido"].to_numpy()])
    n = sum(d in usables for d in ventana)
    frac = n / len(ventana)
    return {"desde": ventana[0].date(), "hasta": fin.date(), "usables": n, "total": len(ventana),
            "fraccion": frac, "alarma": frac < Q["alarma_dias_utiles_min"]}


def marcar_historia(t):
    """historia_incompleta = algún día de los `dias_historia` anteriores falta o fue excluido."""
    fechas = pd.to_datetime(t["dia"], format="%Y_%m_%d")
    usables = set(fechas[~t["excluido"].to_numpy()])
    k = Q["dias_historia"]
    faltan = []
    for f, ex in zip(fechas, t["excluido"]):
        previos = [f - pd.Timedelta(days=i) for i in range(1, k + 1)]
        faltan.append([] if ex else [p.strftime("%d/%m") for p in previos if p not in usables])
    t["dias_previos_faltantes"] = [", ".join(x) for x in faltan]
    t["historia_incompleta"] = [bool(x) for x in faltan]
    return t


def fmt_dia(d):
    return datetime.strptime(d, "%Y_%m_%d").strftime("%d/%m/%Y")


def escribir_registro(t, sistema, cambios):
    ex = t[t["excluido"]]
    al = t[t["alarma_cobertura"]]
    L = ["# Registro de calidad de los datos (generado automáticamente)", "",
         f"Última corrida: {datetime.now():%Y-%m-%d %H:%M}. Lo genera `src/calidad_dias.py`; no editar a mano.", "",
         "## Regla", "",
         f"- Se excluye un día si su % de filas sin observación supera en más de **{Q['max_exceso_pp']} pp** "
         f"la mediana de los **{Q['ventana_dias']} días válidos anteriores** (mínimo {Q['min_dias_referencia']} días de referencia).",
         f"- Alarma por día: cobertura total (real + interpolada) < **{Q['alarma_cobertura_min']:.0%}** en un día incluido.",
         f"- Alarma del sistema: días usables en los últimos {VENTANA_SISTEMA} días de calendario < **{Q['alarma_dias_utiles_min']:.0%}**.", "",
         "## Resumen", "",
         f"- Días con datos: {len(t)} ({fmt_dia(t['dia'].iloc[0])} → {fmt_dia(t['dia'].iloc[-1])}).",
         f"- Días excluidos: {len(ex)}. Días usados en el análisis: {len(t) - len(ex)}.", ""]
    if cambios:
        L += ["**Cambios respecto de la corrida anterior:** " + "; ".join(cambios), ""]
    L += ["## Días excluidos", ""]
    if len(ex):
        L += ["| Día | Sin observación | Referencia | Cobertura total | Cobertura real | Estaciones sin obs. | Horas con apagón |",
              "|---|---|---|---|---|---|---|"]
        for r in ex.itertuples():
            L.append(f"| {fmt_dia(r.dia)} | {r.pct_sin_obs:.1f} % | {r.referencia_pct:.1f} % | {r.cobertura_total:.0%} | "
                     f"{r.cobertura_real:.0%} | {r.estaciones_sin_obs}/{r.estaciones} | {r.horas_apagon} |")
        L += ["", "Nota: los días del 19 al 26/10/2026 corresponden a una falla de observaciones del NWS "
                  "(confirmada por el grupo); su exclusión se acordó con el docente. La regla los detecta sola.", ""]
    else:
        L += ["Ninguno.", ""]
    hi = t[t["historia_incompleta"]]
    L += ["## Días con historia incompleta (se usan, pero se evalúan aparte)", "",
          f"Algún día de los {Q['dias_historia']} previos falta o fue excluido, así que sus rezagos quedan vacíos en buena parte.", ""]
    L += [f"- {fmt_dia(r.dia)}: faltan {r.dias_previos_faltantes}" for r in hi.itertuples()] or ["Ninguno."]
    L += [""]
    L += ["## Alarmas", ""]
    if len(al):
        for r in al.itertuples():
            L.append(f"- **{fmt_dia(r.dia)}**: cobertura total {r.cobertura_total:.0%} (< {Q['alarma_cobertura_min']:.0%}); el día se usa igual.")
    else:
        L.append(f"- Por día: ninguna (todos los días incluidos tienen cobertura total ≥ {Q['alarma_cobertura_min']:.0%}).")
    s = sistema
    L.append(f"- Sistema: {s['usables']} de {s['total']} días usables entre {s['desde']:%d/%m} y {s['hasta']:%d/%m} "
             f"({s['fraccion']:.0%})" + (" → **ALARMA**" if s["alarma"] else " → ok") + ".")
    L += ["", "El detalle de todos los días está en `calidad_por_dia.csv`.", ""]
    (SALIDA / "REGISTRO_CALIDAD.md").write_text("\n".join(L))


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    pares = sorted((RAIZ / CFG["salida"] / "pares").glob("*.parquet"))
    m = pd.DataFrame([medir_dia(p) for p in pares])
    t = aplicar_regla(m)
    sistema = alarma_sistema(t)
    t = marcar_historia(t)

    archivo = SALIDA / "dias_excluidos.json"
    previos = {x["dia"] for x in json.loads(archivo.read_text())} if archivo.exists() else set()
    ex = t[t["excluido"]]
    actuales = set(ex["dia"])
    cambios = ([f"nuevo excluido {fmt_dia(d)}" for d in sorted(actuales - previos)]
               + [f"{'vuelve a incluirse' if d in set(t['dia']) else 'ya no está en los datos'} {fmt_dia(d)}"
                  for d in sorted(previos - actuales)])
    archivo.write_text(json.dumps([{"dia": r.dia, "pct_sin_obs": round(r.pct_sin_obs, 2),
                                    "referencia_pct": round(r.referencia_pct, 2), "motivo": r.motivo}
                                   for r in ex.itertuples()], indent=1, ensure_ascii=False))
    (SALIDA / "historia_incompleta.json").write_text(json.dumps(
        [{"dia": r.dia, "dias_previos_faltantes": r.dias_previos_faltantes}
         for r in t[t["historia_incompleta"]].itertuples()], indent=1, ensure_ascii=False))
    t.round(4).to_csv(SALIDA / "calidad_por_dia.csv", index=False)
    escribir_registro(t, sistema, cambios if archivo.exists() and previos != actuales else [])

    print(f"{len(t)} días medidos | excluidos: {len(ex)} | usados: {len(t) - len(ex)} "
          f"(con historia incompleta: {int(t['historia_incompleta'].sum())})")
    for r in ex.itertuples():
        print(f"AVISO: día {fmt_dia(r.dia)} excluido ({r.pct_sin_obs:.1f} % sin observación, referencia {r.referencia_pct:.1f} %)")
    for c in cambios:
        print("AVISO: cambio en la lista de excluidos:", c)
    for r in t[t["alarma_cobertura"]].itertuples():
        print(f"ALARMA: día {fmt_dia(r.dia)} con cobertura total {r.cobertura_total:.0%} (< {Q['alarma_cobertura_min']:.0%}); se usa igual")
    if sistema["alarma"]:
        print(f"ALARMA: solo {sistema['usables']} de {sistema['total']} días usables entre "
              f"{sistema['desde']:%d/%m} y {sistema['hasta']:%d/%m} ({sistema['fraccion']:.0%})")
    print("registro:", (SALIDA / "REGISTRO_CALIDAD.md").relative_to(RAIZ))


if __name__ == "__main__":
    main()
