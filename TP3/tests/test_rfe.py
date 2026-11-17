import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import modelo_interpretable as MI
import modelo_rfe as R
from ridge_gram import Gram


def datos(colineal=False):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4000, 8)) @ rng.normal(size=(8, 8))
    y = X @ np.array([3.0, -2.0, 1.5, 0.0, 0.7, 0.2, -0.05, 1.0]) + rng.normal(size=4000)
    if colineal:                                         # dos columnas que suman una tercera (como las bases spline)
        X = np.column_stack([X, X[:, 0] + X[:, 1]])
    return StandardScaler().fit_transform(X), y


@pytest.mark.parametrize("colineal", [False, True])
def test_ols_igual_a_sklearn(colineal):
    X, y = datos(colineal)
    coef, b0 = R.ols(Gram.desde(X, y, bloque=900), np.ones(X.shape[1], bool))
    np.testing.assert_allclose(X @ coef + b0, LinearRegression().fit(X, y).predict(X), atol=1e-6)


def test_orden_igual_a_rfe_de_sklearn():
    X, y = datos()
    var = np.array([f"v{i}" for i in range(X.shape[1])])
    pasos, _ = R.rfe([(Gram.desde(X, y), len(X), var)])
    ranking = RFE(LinearRegression(), n_features_to_select=1, step=1).fit(X, y).ranking_   # 1 = la última que queda
    esperado = [f"v{i}" for i in np.argsort(-ranking)]
    assert [p["sale"] for p in pasos] == esperado


def test_variable_de_cubre_todas_las_columnas():
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({c: rng.normal(size=n) for f in MI.TODAS for c in MI.FAMILIAS[f]})
    df["condicion"] = rng.choice(MI.CONDICIONES, n)
    df["noche"] = rng.random(n) < 0.5
    df["horizonte"] = rng.integers(1, 49, n).astype(float)
    df["error_c"] = rng.normal(size=n)
    dis = MI.Disenio(MI.TODAS).fit(df)
    var = R.variable_de(dis)
    assert len(var) == len(dis.nombres)
    esperadas = (set(MI.FAMILIA_DE) - MI.APARTE) | set(R.PROPIAS) | set(MI.CURVAS) | set(MI.LINEALES) | {"condicion"}
    assert set(var) == esperadas
    assert set(var[np.char.startswith(np.array(dis.nombres), "err_ayer_x_")]) == {"err_ayer"}
    assert set(var[np.char.startswith(np.array(dis.nombres), "sesgo_ewma_05")]) == {"sesgo_ewma_05"}


def test_regla_elige_el_mas_chico_tolerado():
    curva = pd.DataFrame({"MAE_pliegue_1": [1.250, 1.249, 1.251, 1.262, 1.300],
                          "MAE_pliegue_2": [1.240, 1.239, 1.241, 1.243, 1.300]}, index=[5, 4, 3, 2, 1])
    n, mejor, tab = R.elegir(curva, umbral_prom=0.005, umbral_pliegue=0.010)
    assert mejor == 4
    assert n == 3                       # el de 2: promedio +0,0085, y el pliegue 1 pasa 0,010
    assert not tab.loc[2, "cumple"]
