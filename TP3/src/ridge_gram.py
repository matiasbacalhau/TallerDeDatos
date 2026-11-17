"""Muchos Ridge (variantes de columnas × penalizaciones α) desde una sola matriz XᵀX por pliegue.

El escalado y los splines se ajustan columna por columna, así que sacar una familia equivale a sacar sus filas y
columnas de XᵀX: el resultado es idéntico a reentrenar sin esas columnas, pero cuesta un sistema lineal chico en
lugar de recorrer otra vez el millón de filas. Igual que sklearn.linear_model.Ridge con fit_intercept=True: se
centran X e y y el intercepto no se penaliza.
"""
import numpy as np


class Gram:
    def __init__(self, xtx, xty, media_x, media_y):
        self.xtx, self.xty, self.media_x, self.media_y = xtx, xty, media_x, media_y

    @classmethod
    def desde(cls, X, y, bloque=200_000):
        n, p = X.shape
        sx, sy = np.zeros(p), 0.0
        xtx, xty = np.zeros((p, p)), np.zeros(p)
        for i in range(0, n, bloque):
            Xb, yb = np.asarray(X[i:i + bloque], np.float64), np.asarray(y[i:i + bloque], np.float64)
            xtx += Xb.T @ Xb
            xty += Xb.T @ yb
            sx += Xb.sum(0)
            sy += yb.sum()
        mx, my = sx / n, sy / n
        return cls(xtx - n * np.outer(mx, mx), xty - n * mx * my, mx, my)

    def resolver(self, mascara, alpha):
        """(coeficientes con 0 fuera de la máscara, intercepto)."""
        m = np.asarray(mascara, bool)
        coef = np.zeros(len(m))
        A = self.xtx[np.ix_(m, m)] + alpha * np.eye(m.sum())
        coef[m] = np.linalg.solve(A, self.xty[m])
        return coef, float(self.media_y - self.media_x @ coef)
