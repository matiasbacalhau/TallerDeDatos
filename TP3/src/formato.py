"""Formato de números y fechas para la salida en pantalla del pipeline (convención argentina).

  n(27818020)      -> '27.818.020'
  dec(1.4743, 3)   -> '1,474'
  pct(0.645)       -> '64,5 %'      (recibe una fracción entre 0 y 1)
  fecha(ts)        -> '29/10/2026'
"""


def n(x):
    return f"{int(x):,}".replace(",", ".")


def dec(x, d=1):
    return f"{x:,.{d}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def pct(frac, d=1):
    return f"{dec(100 * frac, d)} %"


def fecha(ts):
    return f"{ts:%d/%m/%Y}"
