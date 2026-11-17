"""Etapa C: tablas chicas de resultados para las pestañas de comparación del dashboard.

Lee las salidas versionadas de los modelos (salidas/*.txt) y escribe CSV en app/datos/resultados/, para que el
dashboard muestre la búsqueda de hiperparámetros, las versiones y las pruebas de significancia sin necesitar
data/resultados/ (que no está en el repo).

Uso (desde TP3/):  python src/preparar_resultados_dashboard.py
Volver a correrlo después de src/pipeline.py y los modelos, cuando cambien salidas/.
"""
import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SAL = RAIZ / "salidas"
DEST = RAIZ / "app" / "datos" / "resultados"


def numero(s):
    """'1,2455' o '1.991.382' o '-0.0510' → float (formato argentino o inglés)."""
    s = s.strip().replace("%", "")
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    return float(s)


def bloque(texto, titulo):
    """Texto de una sección de salidas/modelos_finales.txt («===== titulo =====»)."""
    m = re.search(rf"===== {titulo} =====\n(.*?)(?:\n===== |\Z)", texto, re.S)
    if not m:
        raise ValueError(f"no encontré la sección {titulo}")
    return m.group(1)


def tablas_mae(texto, version):
    """Las tres tablas «MAE por horizonte» (completa / incompleta / todas) de un log de modelo."""
    filas = []
    segs = {"Días con historia completa": "completa", "Días con historia incompleta": "incompleta",
            "Todos los días de prueba": "todas"}
    for titulo, seg in segs.items():
        m = re.search(rf"{titulo}[^\n]*\n\s+horizonte([^\n]*)\n((?:\s+(?:todos|\d+ h)[^\n]*\n)+)", texto)
        if not m:
            continue
        cols = m.group(1).split()
        for linea in m.group(2).strip().splitlines():
            p = linea.split()
            h = "todos" if p[0] == "todos" else p[0]
            vals = p[2:] if p[1] == "h" else p[1:]
            for c, v in zip(cols, vals):
                filas.append({"version": version, "segmento": seg, "horizonte": h, "modelo": c, "MAE": numero(v)})
    return filas


def alphas(texto, version):
    filas = []
    for a, mae, el in re.findall(r"^\s+(1e[+-]\d+)\s+([\d,]+)(\s+← elegido)?", texto, re.M):
        filas.append({"version": version, "alpha": float(a), "MAE_eleccion": numero(mae), "elegido": bool(el)})
    return filas


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    fin = (SAL / "modelos_finales.txt").read_text()
    ridge5, lgbm = bloque(fin, "modelo_interpretable"), bloque(fin, "modelo_lgbm")
    comp, interp, dash = bloque(fin, "comparar_modelos"), bloque(fin, "interpretar"), bloque(fin, "preparar_dashboard")
    ridge8 = (SAL / "log_ridge_todas_familias.txt").read_text()
    lgbm_log = (SAL / "log_lgbm.txt").read_text()

    # 1. MAE por horizonte y segmento (pliegues de reporte 3–5)
    filas = tablas_mae(lgbm, "final") + [f for f in tablas_mae(ridge8, "ridge_8_familias") if f["modelo"] == "lineal"]
    pd.DataFrame(filas).to_csv(DEST / "mae_horizonte.csv", index=False)

    # 2. búsqueda de α del Ridge (pliegues de elección 1–2), versión final (5 familias) y con las 8
    pd.DataFrame(alphas(ridge5, "Ridge final (5 familias)") + alphas(ridge8, "Ridge con las 8 familias")) \
        .to_csv(DEST / "ridge_alpha.csv", index=False)

    # 3. búsqueda de LightGBM: las 5 mejores combinaciones y el tiempo de cada combinación
    top = re.findall(r"(hojas\d+_regression(?:_l1)?)\s+(\d+)\s+([\d,]+)\s*$", lgbm_log, re.M)
    pd.DataFrame([{"combinacion": c, "num_leaves": int(re.search(r"\d+", c).group()),
                   "objetivo": "L1 (regression_l1)" if c.endswith("_l1") else "L2 (regression)",
                   "arboles": int(a), "MAE_eleccion": numero(m), "puesto": i + 1}
                  for i, (c, a, m) in enumerate(top)]).to_csv(DEST / "lgbm_top5.csv", index=False)
    tiempos = re.findall(r"pliegue (\d) · (hojas\d+_regression(?:_l1)?): (\d+) s", lgbm_log)
    pd.DataFrame([{"pliegue": int(p), "combinacion": c, "segundos": int(s)} for p, c, s in tiempos]) \
        .to_csv(DEST / "lgbm_tiempos.csv", index=False)

    # 4. importancia por familia de LightGBM
    imp = re.findall(r"^\s+(base|pendiente|rezagos_extra|rezagos|sesgo_ewma|vecinos|revision|dinamica)\s+([\d,]+) %",
                     lgbm, re.M)
    pd.DataFrame([{"familia": f, "ganancia_pct": numero(v)} for f, v in imp]) \
        .to_csv(DEST / "lgbm_importancia_familia.csv", index=False)

    # 5. MAE por día de prueba y pruebas de significancia
    dias = re.findall(r"^\s+(\d)\s+(\d{4}_\d\d_\d\d)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\w+)$",
                      comp, re.M)
    pd.DataFrame([{"pliegue": int(p), "dia": d.replace("_", "-"), "n": int(n), "B0": float(b0), "B2": float(b2),
                   "lineal": float(li), "lgbm": float(lg), "historia": h} for p, d, n, b0, b2, li, lg, h in dias]) \
        .to_csv(DEST / "mae_por_dia.csv", index=False)
    pr = re.findall(r"^\s*(todas|completa|incompleta)\s+(\w+)\s+(\w+)\s+(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)"
                    r"\s+(-?[\d.]+)\s+([\d.]+)\s+(\d+)$", comp, re.M)
    pd.DataFrame([{"segmento": s, "A": a, "B": b, "dias": int(d), "dif_media_C": float(x), "IC95_inf": float(lo),
                   "IC95_sup": float(hi), "t": float(t), "p_valor": float(p), "dias_gana_A": int(g)}
                  for s, a, b, d, x, lo, hi, t, p, g in pr]).to_csv(DEST / "pruebas.csv", index=False)

    # 6. interpretación (Etapa B)
    shap = re.findall(r"^(\w+)\s+([\d.]+)$", interp.split("2. SHAP")[1].split("correlación por estación")[0], re.M)
    pd.DataFrame([{"variable": v, "shap_abs_medio_C": float(x)} for v, x in shap]) \
        .to_csv(DEST / "shap_top10.csv", index=False)
    ses = re.findall(r"^(B0|B2|lineal|lgbm)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)$", interp, re.M)
    pd.DataFrame([{"modelo": m, "amplitud_hora": float(a), "desvio_entre_estaciones": float(d), "error_medio": float(e)}
                  for m, a, d, e in ses]).to_csv(DEST / "sesgo_restante.csv", index=False)
    techo = re.findall(r"^(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)$", interp.split("4. correlación")[1], re.M)
    pd.DataFrame([{"horizonte": int(h), "err_ult": float(e), "lineal": float(r), "lgbm": float(g)}
                  for h, e, r, g in techo]).to_csv(DEST / "techo.csv", index=False)
    m = re.search(r"enfriamiento nocturno ([-\d.]+) °C; peso de err_ult a 1/6/24/48 h: ([\d.]+) / ([\d.]+) / "
                  r"([\d.]+) / ([\d.]+)", interp)
    r = re.search(r"lat\+lon r = ([\d.]+) \| sesgos exponenciales r = ([\d.]+)", interp)
    pd.DataFrame([{"enfriamiento_nocturno_C": float(m.group(1)), "peso_err_ult_1h": float(m.group(2)),
                   "peso_err_ult_6h": float(m.group(3)), "peso_err_ult_24h": float(m.group(4)),
                   "peso_err_ult_48h": float(m.group(5)), "r_latlon_sesgo": float(r.group(1)),
                   "r_ewma_sesgo": float(r.group(2))}]).to_csv(DEST / "ridge_lectura.csv", index=False)

    # 7. cobertura de la banda P10–P90
    cob = re.findall(r"^\s*(lineal|lgbm)\s+([\d.]+)\s+(\d+)$", dash, re.M)
    pd.DataFrame([{"modelo": m, "cobertura": float(c), "n": int(n)} for m, c, n in cob]) \
        .to_csv(DEST / "cobertura_global.csv", index=False)

    # 8. datos y validación (salidas/salida_pipeline.txt): pliegues, días excluidos y baselines B0–B3
    pipe = (SAL / "salida_pipeline.txt").read_text()
    pl = re.findall(r"^\s+(\d)\s+(elige|reporta)\s+(\d\d/\d\d) → (\d\d/\d\d) \((\d+) días\)\s+(\d+) días\s+"
                    r"(\d\d/\d\d) → (\d\d/\d\d) \((.*?)\)$", pipe, re.M)
    pd.DataFrame([{"pliegue": int(p), "uso": u, "entrena_desde": a, "entrena_hasta": b, "dias_entrena": int(n),
                   "separacion_dias": int(s), "prueba_desde": c, "prueba_hasta": d, "prueba": t}
                  for p, u, a, b, n, s, c, d, t in pl]).to_csv(DEST / "pliegues.csv", index=False)
    ex = re.findall(r"^\s+(\d\d/\d\d/\d{4})\s+([\d,]+) %\s+([\d,]+) %\s+([\d,]+) %$", pipe, re.M)
    pd.DataFrame([{"dia": d, "sin_observacion_pct": numero(a), "referencia_pct": numero(b), "limite_pct": numero(c)}
                  for d, a, b, c in ex]).to_csv(DEST / "dias_excluidos.csv", index=False)
    filas = []
    for letra, seg in (("A", "completa"), ("B", "incompleta"), ("C", "todas")):
        m = re.search(rf"^{letra}\. .*?\n\s+horizonte\s+B0\s+B1\s+B2\s+B3\s+mejor\n(.*?)\n\n", pipe, re.M | re.S)
        for linea in m.group(1).splitlines():
            p = linea.split()
            if p[0] == "MAE":
                h = p[1] if p[1] == "todos" else p[1]
                vals = p[3:7] if p[2] == "h" else p[2:6]
                metrica = "MAE"
            else:
                h, metrica, vals = "todos", p[0], p[2:6]
            for b, x in zip(("B0", "B1", "B2", "B3"), vals):
                filas.append({"segmento": seg, "metrica": metrica, "horizonte": h, "baseline": b, "valor": numero(x)})
    pd.DataFrame(filas).to_csv(DEST / "baselines.csv", index=False)

    for p in sorted(DEST.glob("*.csv")):
        print(f"{p.name:32s} {len(pd.read_csv(p)):4d} filas")


if __name__ == "__main__":
    main()
