import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import modelo_interpretable as MI


def sintetico(n=2000, semilla=1):
    rng = np.random.default_rng(semilla)
    d = {}
    for cols in MI.FAMILIAS.values():
        for c in cols:
            v = rng.normal(size=n)
            v[rng.random(n) < 0.1] = np.nan
            d[c] = v
    ang = rng.uniform(0, 2 * np.pi, n)
    d.update(hora_solar_sin=np.sin(ang), hora_solar_cos=np.cos(ang), horizonte=rng.integers(1, 49, n),
             noche=rng.random(n) < 0.5, viento_calmo=rng.random(n) < 0.2, viento_vel=rng.uniform(0, 30, n),
             condicion=rng.choice(MI.CONDICIONES, n), error_c=rng.normal(size=n))
    return pd.DataFrame(d)


def test_todas_las_familias_tienen_columnas():
    dis = MI.Disenio(MI.TODAS).fit(sintetico())
    assert set(dis.familia_de) == set(MI.TODAS)


def test_sacar_vecinos():
    dis = MI.Disenio([f for f in MI.TODAS if f != "vecinos"]).fit(sintetico())
    assert not any(n.startswith("vec_") for n in dis.nombres)
    assert "vecinos" not in dis.familia_de


def test_sacar_base_mantiene_ponderacion_por_horizonte():
    dis = MI.Disenio([f for f in MI.TODAS if f != "base"]).fit(sintetico())
    assert any(n.startswith("err_ult_x_h_s") for n in dis.nombres)
    assert not any(n.startswith(("cond_", "hora_s")) for n in dis.nombres)
    assert "horizonte" not in dis.nombres


def test_contribucion_suma_prediccion():
    df = sintetico()
    dis = MI.Disenio(MI.TODAS).fit(df)
    X = dis.transform(df)
    m = Ridge(alpha=1.0).fit(X, df["error_c"])
    C = MI.contribuciones(dis, m.coef_, X)
    np.testing.assert_allclose(C.sum(axis=1) + m.intercept_, m.predict(X), atol=1e-4)
    s = MI.contribucion_por_familia(dis, m.coef_, X)
    assert set(s.index) == set(MI.TODAS) and (s >= 0).all()


def test_hiperparametros_viejos_no_sirven(tmp_path):
    ruta = tmp_path / "hp.json"
    ruta.write_text(json.dumps({"sesgo": "sesgo_ewma_10", "ridge_alpha": 1.0}))
    assert MI.hiperparametros_vigentes(ruta, MI.TODAS) is None
    ruta.write_text(json.dumps({"familias": MI.TODAS, "ridge_alpha": 1e5}))
    assert MI.hiperparametros_vigentes(ruta, MI.TODAS)["ridge_alpha"] == 1e5
    assert MI.hiperparametros_vigentes(ruta, MI.TODAS[:3]) is None
    assert MI.hiperparametros_vigentes(tmp_path / "no.json", MI.TODAS) is None
