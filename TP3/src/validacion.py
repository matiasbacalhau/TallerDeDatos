"""Validación temporal walk-forward (expanding window) con gap.

Los cortes se hacen por DÍA DE LA HORA OBJETIVO (target_time_utc), nunca por filas
sueltas: cada hora objetivo aparece hasta 48 veces (una por horizonte) con la misma
observación, así que mezclar filas pondría "la respuesta" a ambos lados del corte.

Pliegues ANCLADOS al calendario (acordado el 17/11/2026): los bloques de prueba se cuentan desde una fecha
fija (validacion.ancla_pruebas), en grupos de dias_prueba días usables. Un día nuevo no cambia los pliegues
anteriores; cuando se completa un bloque aparece un pliegue nuevo al final. Los primeros n_pliegues_eleccion
pliegues eligen hiperparámetros (una sola vez); los siguientes solo se evalúan.

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
import json
from pathlib import Path

import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((RAIZ / "config.yaml").read_text())["validacion"]


def dias_excluidos():
    """Días que el control de calidad (src/calidad_dias.py) sacó del análisis."""
    archivo = RAIZ / "data" / "calidad" / "dias_excluidos.json"
    if not archivo.exists():
        return set()
    return {x["dia"] for x in json.loads(archivo.read_text())}


def dias_historia_incompleta():
    """Días válidos cuyos días previos faltan o fueron excluidos (rezagos degradados)."""
    archivo = RAIZ / "data" / "calidad" / "historia_incompleta.json"
    if not archivo.exists():
        return set()
    return {x["dia"] for x in json.loads(archivo.read_text())}


def dias_disponibles(carpeta=None, excluir=True):
    """Días objetivo disponibles (un parquet por día en data/pares/), sin los días excluidos
    por calidad. Todo lo que entrena o evalúa pasa por acá, así la exclusión es automática."""
    carpeta = Path(carpeta) if carpeta else RAIZ / "data" / "pares"
    fuera = dias_excluidos() if excluir else set()
    return pd.DatetimeIndex(sorted(pd.to_datetime(p.stem, format="%Y_%m_%d")
                                   for p in carpeta.glob("*.parquet") if p.stem not in fuera))


def _bloques(dias, hasta=None, ancla=None, dias_prueba=None):
    """Días usables desde la ancla (y hasta `hasta`) cortados en bloques de dias_prueba; el resto queda en espera."""
    ancla = pd.Timestamp(ancla or CFG["ancla_pruebas"])
    dias_prueba = dias_prueba or CFG["dias_prueba"]
    dias = pd.DatetimeIndex(sorted(set(dias)))
    if hasta is not None:
        dias = dias[dias <= pd.Timestamp(hasta)]
    candidatos = dias[dias >= ancla]
    n = len(candidatos) // dias_prueba
    return dias, [candidatos[i * dias_prueba:(i + 1) * dias_prueba] for i in range(n)], candidatos[n * dias_prueba:]


def dias_en_espera(dias, hasta=None, ancla=None, dias_prueba=None):
    """Días usables posteriores al último bloque completo: esperan a que lleguen más días."""
    return _bloques(dias, hasta, ancla, dias_prueba)[2]


def pliegues_walk_forward(dias, hasta=None, ancla=None, dias_prueba=None, gap_dias=None, dias_min_entrenamiento=None):
    """Devuelve una lista de (dias_entrenamiento, dias_prueba) como DatetimeIndex.

    Pliegues ANCLADOS al calendario: los bloques de prueba son grupos consecutivos de dias_prueba días usables
    desde la fecha ancla. Un día nuevo no cambia los pliegues anteriores: solo agrega uno cuando completa un bloque.
    """
    gap_dias = CFG["gap_dias"] if gap_dias is None else gap_dias
    dias_min = dias_min_entrenamiento or CFG["dias_min_entrenamiento"]
    dias, bloques, _ = _bloques(dias, hasta, ancla, dias_prueba)
    pliegues = []
    for k, prueba in enumerate(bloques):
        # gap en días de CALENDARIO (puede haber días sin archivo): el último día de entrenamiento
        # queda al menos gap_dias+1 días antes del primer día de prueba
        entrena = dias[dias <= prueba.min() - pd.Timedelta(days=gap_dias + 1)]
        if len(entrena) < dias_min:
            raise ValueError(f"Pliegue {k+1}: solo {len(entrena)} días de entrenamiento "
                             f"(mínimo {dias_min}). Mover la ancla más adelante.")
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
    from formato import fecha
    d = dias_disponibles()
    pl = pliegues_walk_forward(d)
    verificar(pl, CFG["gap_dias"])
    g, ne = CFG["gap_dias"], CFG["n_pliegues_eleccion"]
    print("Arma los cortes para evaluar los modelos como se usarían de verdad: cada modelo entrena con días\n"
          f"pasados y se prueba con días posteriores. Entre ambos quedan al menos {g} días sin usar para que la prueba no comparta\n"
          "la memoria del error (unas 6 h) ni el mismo régimen de tiempo con el final del entrenamiento.\n"
          f"Los bloques de prueba se cuentan desde el {fecha(pd.Timestamp(CFG['ancla_pruebas']))}, de a {CFG['dias_prueba']} días "
          "usables: un día nuevo no cambia los pliegues\nanteriores. Los primeros "
          f"{ne} pliegues eligen los hiperparámetros (una sola vez); los demás solo se evalúan.\n")
    print(f"Días disponibles (sin los excluidos): {len(d)}, del {d.min():%d/%m/%Y} al {d.max():%d/%m/%Y}. Pliegues: {len(pl)}.\n")
    print("  pliegue   uso       entrena con               separación     prueba con")
    for k, (e, p) in enumerate(pl, 1):
        hueco = (p.max() - p.min()).days + 1 > len(p)
        print(f"  {k:<9} {'elige' if k <= ne else 'reporta':<9} {e.min():%d/%m} → {e.max():%d/%m} ({len(e):>2} días)   "
              f"{(p.min() - e.max()).days - 1} días         {p.min():%d/%m} → {p.max():%d/%m} ({len(p)} días"
              + (", salteando días excluidos o sin archivo)" if hueco else ")"))
    espera = dias_en_espera(d)
    print(f"\nDías en espera de completar un bloque: "
          + (", ".join(f"{x:%d/%m}" for x in espera) + f" ({len(espera)} de {CFG['dias_prueba']})" if len(espera) else "ninguno"))
    print("\nControles: el entrenamiento siempre termina antes de la prueba, con la separación pedida, y los "
          "bloques de prueba no se pisan → OK")
