import pandas as pd
import pytest

import experimentos as E
from validacion import pliegues_walk_forward

BASE = pd.DatetimeIndex(list(pd.date_range("2026-09-10", "2026-10-14"))
                        + [pd.Timestamp(f"2026-10-{d}") for d in (16, 17, 18, 21, 29)]
                        + list(pd.date_range("2026-11-02", "2026-11-09")))


@pytest.mark.parametrize("prom, maximo, esperado", [
    (0.004, 0.009, True), (0.006, 0.006, False), (0.003, 0.012, False), (-0.010, -0.002, True), (0.005, 0.010, True)])
def test_regla(prom, maximo, esperado):
    assert E.aplicar_regla(prom, maximo, umbral_prom=0.005, umbral_pliegue=0.010) is esperado


def test_confirmar_recorte_devuelve_la_mas_costosa():
    def evaluar(sacar):                        # empeora 0,008 solo si se sacan A y B juntas
        return 0.008 if {"A", "B"} <= set(sacar) else 0.001
    assert E.confirmar_recorte(["A", "B"], {"A": 0.004, "B": 0.002}, evaluar, umbral_prom=0.005) == ["B"]


def test_confirmar_recorte_sin_problema_deja_todo():
    assert E.confirmar_recorte(["A", "B"], {"A": 0.004, "B": 0.002}, lambda s: 0.001, umbral_prom=0.005) == ["A", "B"]


def test_dias_exploracion_no_toca_reporte():
    pl = pliegues_walk_forward(BASE, hasta=pd.Timestamp("2026-11-09"))
    primer_reporte = pl[E.CFG_VAL["n_pliegues_eleccion"]][1].min()
    d = E.dias_exploracion(BASE, hasta=pd.Timestamp("2026-11-09"))
    assert d.max() < primer_reporte
    assert d.max() == pl[E.CFG_VAL["n_pliegues_eleccion"] - 1][1].max()
