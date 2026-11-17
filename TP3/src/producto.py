"""Modelo PRODUCTO: entrena el modelo que eligió el grupo con todos los días disponibles y lo deja listo para usar.

La parte experimental de cada tipo de modelo está en src/modelos/ (ridge, lgbm, ols_rfe): ahí se eligen los
hiperparámetros y se mide el error en los pliegues de reporte. Cuál de ellos es el producto lo dice config.yaml
(producto.modelo) y se decide con notebooks/comparacion_modelos.ipynb. Este script no elige nada:
  1. lee los hiperparámetros congelados del modelo elegido (si faltan o son viejos, se detiene y dice qué correr);
  2. lo entrena con todos los días usables que tienen variables (data/features/);
  3. guarda el modelo y su ficha en data/producto/ (modelo.pkl, ficha.json).

Es idempotente: si el modelo, sus hiperparámetros, el código y los datos no cambiaron, no vuelve a entrenar.
Para usarlo:  import producto; c = producto.predecir(df)   (df con las columnas de data/features/)

Uso:  .venv/bin/python src/producto.py
"""
import json
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd

import cache
import modelos
from modelado import CFG, FEAT_DIR, RAIZ
from validacion import dias_disponibles

SALIDA = RAIZ / "data" / "producto"


def elegido():
    return CFG["producto"]["modelo"]


def dias_de_entrenamiento():
    """Días usables (sin los excluidos por calidad) que ya tienen variables calculadas."""
    return pd.DatetimeIndex([d for d in dias_disponibles() if (FEAT_DIR / f"{d:%Y_%m_%d}.parquet").exists()])


def metricas_reporte(mod):
    """MAE del modelo en los pliegues de reporte (todos los días de prueba), leído de su CSV de resultados."""
    if not mod.REPORTE.exists():
        return None
    R = pd.read_csv(mod.REPORTE).astype({"horizonte": str})
    R = R[(R.segmento == "todas") & (R.horizonte == "todos")]
    t = R.groupby("modelo").apply(lambda g: np.average(g.MAE, weights=g.n), include_groups=False)
    return {"pliegues": sorted(int(p) for p in R.pliegue.unique()), "MAE_C": {m: round(float(v), 4) for m, v in t.items()}}


def main():
    nombre = elegido()
    mod = modelos.cargar(nombre)
    hp = mod.hiperparametros_producto()
    dias = dias_de_entrenamiento()
    clave = cache.clave(modelo=nombre, hp=hp, codigo=cache.huella_archivos([mod.__file__, RAIZ / "src" / "modelado.py"]),
                        dias=cache.fechas(dias), huellas=cache.huella_dias(dias))
    ruta_ficha = SALIDA / "ficha.json"
    if ruta_ficha.exists() and json.loads(ruta_ficha.read_text()).get("clave") == clave:
        print(f"Producto ({nombre}) al día: mismo modelo, hiperparámetros, código y datos. No se reentrena.")
        return

    print(f"Producto: {nombre} (config.yaml) · {len(dias)} días ({dias.min():%d/%m} → {dias.max():%d/%m}) · "
          f"hiperparámetros: {hp}", flush=True)
    t0 = time.time()
    modelo, filas = mod.entrenar_producto(dias, hp)
    SALIDA.mkdir(parents=True, exist_ok=True)
    with open(SALIDA / "modelo.pkl", "wb") as f:
        pickle.dump({"modelo_tipo": nombre, "modelo": modelo}, f)
    ficha = {"modelo": nombre, "columna": mod.COLUMNA, "hiperparametros": hp, "entrenado": datetime.now().isoformat(timespec="seconds"),
             "dias": {"primero": f"{dias.min():%Y-%m-%d}", "ultimo": f"{dias.max():%Y-%m-%d}", "cantidad": len(dias)},
             "filas_entrenamiento": filas, "reporte": metricas_reporte(mod), "clave": clave}
    ruta_ficha.write_text(json.dumps(ficha, indent=1, ensure_ascii=False))
    print(f"Entrenado con {filas:,} filas en {time.time() - t0:.0f} s. Guardado en {SALIDA.relative_to(RAIZ)}/")
    if ficha["reporte"]:
        print(f"MAE en los pliegues de reporte {ficha['reporte']['pliegues']} (°C): {ficha['reporte']['MAE_C']}")


def predecir(df):
    """Corrección predicha (°C) por el modelo producto guardado. Pronóstico corregido = pronóstico NWS + corrección."""
    with open(SALIDA / "modelo.pkl", "rb") as f:
        guardado = pickle.load(f)
    return modelos.cargar(guardado["modelo_tipo"]).predecir_producto(guardado["modelo"], df)


if __name__ == "__main__":
    main()
