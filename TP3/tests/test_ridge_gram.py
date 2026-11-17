import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ridge_gram import Gram


def datos():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 30)) @ rng.normal(size=(30, 30))      # columnas correlacionadas
    y = X[:, :5].sum(1) + rng.normal(size=5000) + 3.0
    return StandardScaler().fit_transform(X), y


def test_igual_a_sklearn():
    X, y = datos()
    g = Gram.desde(X, y, bloque=700)
    for a in (1.0, 1e3, 1e5):
        coef, b0 = g.resolver(np.ones(30, bool), a)
        m = Ridge(alpha=a).fit(X, y)
        np.testing.assert_allclose(coef, m.coef_, atol=1e-6)
        assert abs(b0 - m.intercept_) < 1e-6


def test_mascara_igual_a_reentrenar_sin_columnas():
    X, y = datos()
    mascara = np.ones(30, bool)
    mascara[[1, 4, 7, 10, 13, 16, 19, 22, 25, 28]] = False
    coef, _ = Gram.desde(X, y).resolver(mascara, 100.0)
    m = Ridge(alpha=100.0).fit(X[:, mascara], y)
    np.testing.assert_allclose(coef[mascara], m.coef_, atol=1e-6)
    assert (coef[~mascara] == 0).all()
