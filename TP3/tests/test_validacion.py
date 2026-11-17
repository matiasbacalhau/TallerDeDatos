import pandas as pd

from validacion import dias_en_espera, pliegues_walk_forward, verificar

ANCLA = pd.Timestamp("2026-09-30")
BASE = pd.DatetimeIndex(list(pd.date_range("2026-09-10", "2026-10-14"))
                        + [pd.Timestamp(f"2026-10-{d}") for d in (16, 17, 18, 21, 29)])


def pl(dias, **kw):
    return pliegues_walk_forward(dias, ancla=ANCLA, dias_prueba=5, gap_dias=2, dias_min_entrenamiento=10, **kw)


def test_pliegues_iguales_a_los_actuales():
    p = pl(BASE)
    assert len(p) == 4
    assert [(te.min(), te.max()) for _, te in p[:3]] == [
        (pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-04")),
        (pd.Timestamp("2026-10-05"), pd.Timestamp("2026-10-09")),
        (pd.Timestamp("2026-10-10"), pd.Timestamp("2026-10-14"))]
    assert list(p[3][1].day) == [16, 17, 18, 21, 29]


def test_dia_nuevo_no_cambia_pliegues_previos():
    antes = pl(BASE)
    dias = BASE.append(pd.DatetimeIndex(["2026-11-02"]))
    despues = pl(dias)
    assert len(despues) == len(antes)
    for (a_tr, a_te), (d_tr, d_te) in zip(antes, despues):
        assert a_tr.equals(d_tr) and a_te.equals(d_te)
    assert list(dias_en_espera(dias, ancla=ANCLA, dias_prueba=5)) == [pd.Timestamp("2026-11-02")]


def test_bloque_completo_agrega_pliegue():
    dias = BASE.append(pd.date_range("2026-11-02", "2026-11-06"))
    p = pl(dias)
    assert len(p) == 5
    tr, te = p[4]
    assert (te.min(), te.max()) == (pd.Timestamp("2026-11-02"), pd.Timestamp("2026-11-06"))
    assert tr.max() <= pd.Timestamp("2026-10-30")
    assert len(dias_en_espera(dias, ancla=ANCLA, dias_prueba=5)) == 0


def test_hasta_recorta():
    assert len(pl(BASE, hasta=pd.Timestamp("2026-10-14"))) == 3


def test_verificar_pasa():
    for dias in (BASE, BASE.append(pd.date_range("2026-11-02", "2026-11-06"))):
        verificar(pl(dias), 2)
