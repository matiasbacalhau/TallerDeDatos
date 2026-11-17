import pytest

import modelos
import producto


@pytest.mark.parametrize("nombre", modelos.DISPONIBLES)
def test_cada_modelo_tiene_la_interfaz_de_producto(nombre):
    m = modelos.cargar(nombre)
    for atributo in ("COLUMNA", "REPORTE", "hiperparametros_producto", "entrenar_producto", "predecir_producto"):
        assert hasattr(m, atributo), f"{nombre} no tiene {atributo}"


def test_modelo_desconocido():
    with pytest.raises(ValueError):
        modelos.cargar("xgboost")


def test_el_producto_de_config_existe():
    assert producto.elegido() in modelos.DISPONIBLES
