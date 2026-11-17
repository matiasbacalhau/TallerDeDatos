"""Caché de resultados por pliegue: si los datos y la configuración no cambiaron, no se vuelve a entrenar.

La huella de cada día es el sha256 del CONTENIDO de su parquet en data/features/ (features.py reescribe todos los
días en cada corrida, así que la fecha de modificación no sirve). La clave junta las huellas con todo lo demás que
determina el resultado (días, columnas, hiperparámetros, versión del modelo): si algo cambia, cambia la clave.
"""
from pathlib import Path
import hashlib
import json
import pickle

RAIZ = Path(__file__).resolve().parents[1]
FEAT_DIR = RAIZ / "data" / "features"
CARPETA = RAIZ / "data" / "cache"


def huella_dias(dias, carpeta=FEAT_DIR):
    huellas = {}
    for d in dias:
        with open(Path(carpeta) / f"{d:%Y_%m_%d}.parquet", "rb") as f:
            huellas[f"{d:%Y_%m_%d}"] = hashlib.file_digest(f, "sha256").hexdigest()
    return huellas


def clave(**partes):
    return hashlib.sha256(json.dumps(partes, sort_keys=True, default=str).encode()).hexdigest()


def leer(tipo, clave):
    archivo = CARPETA / tipo / f"{clave}.pkl"
    return pickle.loads(archivo.read_bytes()) if archivo.exists() else None


def guardar(tipo, clave, objeto):
    (CARPETA / tipo).mkdir(parents=True, exist_ok=True)
    (CARPETA / tipo / f"{clave}.pkl").write_bytes(pickle.dumps(objeto))
