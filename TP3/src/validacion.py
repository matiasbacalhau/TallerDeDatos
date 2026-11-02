"""Validación temporal walk-forward (expanding window) con gap.

Los cortes se hacen por DÍA DE LA HORA OBJETIVO (target_time_utc), nunca por filas
sueltas: cada hora objetivo aparece hasta 48 veces (una por horizonte) con la misma
observación, así que mezclar filas pondría "la respuesta" a ambos lados del corte.

Esquema (ejemplo con 4 pliegues, bloques de 5 días y gap de 2):

    días:   1 ............................................ N
    pliegue 1: [entrena ........] gap [prueba]
    pliegue 2: [entrena ................] gap [prueba]
    pliegue 3: [entrena ........................] gap [prueba]
    pliegue 4: [entrena ................................] gap [prueba]

El entrenamiento siempre es el pasado y crece; los bloques de prueba son
consecutivos y no se pisan. El gap evita que la última hora de entrenamiento y
la primera de prueba compartan la memoria del error (~6 h) o el mismo régimen
de tiempo.
"""
from pathlib import Path

import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((RAIZ / "config.yaml").read_text())["validacion"]


def dias_disponibles(carpeta=None):
    """Días objetivo disponibles (un parquet por día en data/pares/)."""
    carpeta = Path(carpeta) if carpeta else RAIZ / "data" / "pares"
    return pd.DatetimeIndex(sorted(pd.to_datetime(p.stem, format="%Y_%m_%d") for p in carpeta.glob("*.parquet")))


def pliegues_walk_forward(dias, n_pliegues=None, dias_prueba=None, gap_dias=None, dias_min_entrenamiento=None):
    """Devuelve una lista de (dias_entrenamiento, dias_prueba) como DatetimeIndex.

    Los bloques de prueba son los últimos n_pliegues * dias_prueba días, uno tras otro.
    """
    n_pliegues = n_pliegues or CFG["n_pliegues"]
    dias_prueba = dias_prueba or CFG["dias_prueba"]
    gap_dias = CFG["gap_dias"] if gap_dias is None else gap_dias
    dias_min = dias_min_entrenamiento or CFG["dias_min_entrenamiento"]

    dias = pd.DatetimeIndex(sorted(set(dias)))
    inicio_pruebas = len(dias) - n_pliegues * dias_prueba
    pliegues = []
    for k in range(n_pliegues):
        a = inicio_pruebas + k * dias_prueba           # primer día de prueba (posición)
        prueba = dias[a:a + dias_prueba]
        entrena = dias[:max(a - gap_dias, 0)]           # todo el pasado salvo el gap
        if len(entrena) < dias_min:
            raise ValueError(f"Pliegue {k+1}: solo {len(entrena)} días de entrenamiento "
                             f"(mínimo {dias_min}). Bajar n_pliegues/dias_prueba o esperar más datos.")
        pliegues.append((entrena, prueba))
    return pliegues


def verificar(pliegues, gap_dias):
    """Controles que no pueden fallar: sin solapes, con gap, pruebas consecutivas."""
    for entrena, prueba in pliegues:
        assert entrena.max() < prueba.min(), "el entrenamiento llega al período de prueba"
        assert (prueba.min() - entrena.max()).days > gap_dias, "falta el gap"
    for (_, p1), (_, p2) in zip(pliegues, pliegues[1:]):
        assert p1.max() < p2.min(), "bloques de prueba solapados"


if __name__ == "__main__":
    d = dias_disponibles()
    pl = pliegues_walk_forward(d)
    verificar(pl, CFG["gap_dias"])
    print(f"{len(d)} días ({d.min().date()} → {d.max().date()}) | {len(pl)} pliegues")
    for k, (e, p) in enumerate(pl, 1):
        print(f"pliegue {k}: entrena {e.min().date()} → {e.max().date()} ({len(e)} d) | "
              f"gap {CFG['gap_dias']} d | prueba {p.min().date()} → {p.max().date()} ({len(p)} d)")
    print("controles OK")
