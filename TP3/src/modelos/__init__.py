"""Modelos del TP3. Cada archivo es la parte EXPERIMENTAL de un tipo de modelo: elige sus hiperparámetros en los
pliegues de elección (una vez, quedan congelados) y reporta en los pliegues de reporte.

  ridge     Ridge con curvas (caja de cristal), familias elegidas por ablación      src/modelos/ridge.py
  lgbm      LightGBM con las 8 familias (minimiza el error)                         src/modelos/lgbm.py
  ols_rfe   OLS sin penalización, variables elegidas por RFE (caja de cristal)      src/modelos/ols_rfe.py

Cuál es el PRODUCTO lo dice config.yaml (producto.modelo) y lo entrena src/producto.py. Para eso cada módulo
expone la misma interfaz:
  COLUMNA                          nombre del modelo en evaluar() y en los CSV de resultados
  REPORTE                          CSV con sus métricas en los pliegues de reporte
  hiperparametros_producto()       los congelados; error si faltan o se eligieron con otros pliegues
  entrenar_producto(dias, hp)      -> (modelo serializable, filas de entrenamiento)
  predecir_producto(modelo, df)    -> corrección predicha (°C) para cada fila de df
"""
import importlib

DISPONIBLES = ("ridge", "lgbm", "ols_rfe")


def cargar(nombre):
    if nombre not in DISPONIBLES:
        raise ValueError(f"modelo desconocido: {nombre!r}; opciones: {', '.join(DISPONIBLES)}")
    return importlib.import_module(f"modelos.{nombre}")
