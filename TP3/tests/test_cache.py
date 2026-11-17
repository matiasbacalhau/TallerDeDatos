import os

import pandas as pd

import cache

DIAS = [pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-02")]


def escribir(carpeta, valor=1.0):
    for d in DIAS:
        pd.DataFrame({"x": [valor, 2.0, 3.0], "s": ["a", "b", "c"]}).to_parquet(carpeta / f"{d:%Y_%m_%d}.parquet", index=False)


def clave_de(carpeta, dias=DIAS):
    return cache.clave(dias=[f"{d:%Y_%m_%d}" for d in dias], huellas=cache.huella_dias(dias, carpeta), alpha=1.0)


def test_misma_entrada_misma_clave(tmp_path):
    escribir(tmp_path)
    assert clave_de(tmp_path) == clave_de(tmp_path)


def test_cambia_contenido_cambia_clave(tmp_path):
    escribir(tmp_path)
    antes = clave_de(tmp_path)
    escribir(tmp_path, valor=9.0)
    assert clave_de(tmp_path) != antes


def test_reescribir_igual_no_cambia_clave(tmp_path):
    escribir(tmp_path)
    antes = clave_de(tmp_path)
    escribir(tmp_path)                                   # features.py reescribe todos los días en cada corrida
    for d in DIAS:
        os.utime(tmp_path / f"{d:%Y_%m_%d}.parquet", (1e9, 1e9))
    assert clave_de(tmp_path) == antes


def test_cambia_lista_de_dias_cambia_clave(tmp_path):
    escribir(tmp_path)
    assert clave_de(tmp_path, DIAS[:1]) != clave_de(tmp_path)


def test_guardar_leer(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CARPETA", tmp_path / "cache")
    cache.guardar("ridge", "abc", {"filas": [1, 2]})
    assert cache.leer("ridge", "abc") == {"filas": [1, 2]}


def test_leer_inexistente_devuelve_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CARPETA", tmp_path / "cache")
    assert cache.leer("ridge", "no-existe") is None


def test_huella_archivos_depende_del_contenido(tmp_path):
    a = tmp_path / "a.py"
    a.write_text("x = 1\n")
    h = cache.huella_archivos([a])
    a.write_text("x = 1\n")
    assert cache.huella_archivos([a]) == h
    a.write_text("x = 2\n")
    assert cache.huella_archivos([a]) != h
