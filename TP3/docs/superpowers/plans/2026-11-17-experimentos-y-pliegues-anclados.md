# Experimentación por familias y pliegues anclados — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pasar el producto a pliegues anclados al calendario con caché por pliegue. Agregar una experimentación
por familias (1–8) que decida, con una regla fijada de antemano, qué familias usa el Ridge. Entrenar los modelos
finales: Ridge con las familias elegidas y LightGBM con las 8, comparados contra B0–B3.

**Architecture:** `validacion.py` define los pliegues desde una ancla fija. `cache.py` guarda cada resultado
(modelo × pliegue) bajo una huella del contenido de sus datos. `modelo_interpretable.py` deja de depender de flags
y arma la matriz para cualquier lista de familias. `ridge_gram.py` resuelve muchos Ridge (variantes × α) desde una
sola matriz XᵀX por pliegue, así la ablación es casi gratis. `experimentos.py` tiene la lógica de la
experimentación y `notebooks/experimentos.ipynb` la presenta, la corre y exporta los resultados.

**Tech Stack:** Python 3, pandas 3.1, numpy 2.4, scikit-learn 1.9.1, LightGBM 4.7, matplotlib, Jupyter (nbconvert), pytest.

**Spec:** `TP3/docs/superpowers/specs/2026-11-17-experimentos-y-pliegues-anclados-design.md`

## Global Constraints

- Todos los comandos se corren desde `TP3/` con `.venv/bin/python`.
- Parámetros en `config.yaml`, nada hardcodeado: `validacion.ancla_pruebas: 2026-09-30`,
  `validacion.n_pliegues_eleccion: 2`, `dias_prueba: 5`, `gap_dias: 2`, `dias_min_entrenamiento: 10`,
  `experimentos.fecha_corte: 2026-11-09`, `experimentos.umbral_promedio_c: 0.005`, `experimentos.umbral_pliegue_c: 0.010`.
- Regla de recorte: una familia se saca si sin ella el MAE promedio de los pliegues de elección no empeora en más de **0,005 °C** y en ningún pliegue de elección empeora en más de **0,010 °C**.
- El criterio de MAE es siempre el segmento **historia completa**, sobre filas con observación real (igual que hoy).
- LightGBM se entrena siempre con las 8 familias. Elastic Net queda fuera.
- La salida en pantalla se lee como informe: cada número con su significado y su unidad, y `formato.py` para números y fechas.
- No borrar ni reescribir celdas de `notebooks/comparacion_modelos.ipynb`: lo nuevo va en celdas aparte.
- DECISIONES.md registra solo acuerdos. Idioma: español.
- Commits frecuentes en la rama `tp3-experimentos`, con la línea `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Bloque final incompleto** (07–09/11, 3 días usables): no puede formar un pliegue y tiene que aparecer en pantalla como "esperando días". Lo cubre un test de la Tarea 1.
2. **Un día pasado que se excluye o cambia después** (por ejemplo, cuando se cierra un hueco interpolado): la caché de los pliegues que lo usan tiene que invalidarse. Lo cubre un test de la Tarea 2.
3. **`features.py` reescribe todos los parquet en cada corrida:** la huella tiene que depender del contenido y no de la fecha de modificación, o la caché no sirve nunca. Lo cubre un test de la Tarea 2.
4. **Hiperparámetros viejos** (`*_hiperparametros.json` del esquema anterior, con clave `sesgo` y sin `familias`): no pueden reutilizarse en silencio. Lo cubre un test de la Tarea 3.
5. **Sacar la familia `base`** se lleva `horizonte` como columna, pero la base spline del horizonte que pondera `err_ult` tiene que seguir existiendo. Lo cubre un test de la Tarea 3.

---

### Tarea 0: Datos nuevos y entorno de tests

**Files:**
- Modify: `TP3/requirements.txt` (agregar `pytest==8.3.5`)
- Create: `TP3/tests/conftest.py` (agrega `TP3/src` a `sys.path`)

- [ ] **Paso 1:** Esperar a que termine la corrida de `src/pipeline.py` lanzada el 17/11, que escribe en `salidas/salida_pipeline.txt`. Confirmar que la última línea dice "Pipeline completo" y anotar qué días del 02/11–09/11 quedaron excluidos (paso 2 de la salida).
- [ ] **Paso 2:** `.venv/bin/pip install pytest==8.3.5` y crear `conftest.py`. Correr `.venv/bin/python -m pytest -q` y esperar "no tests ran".
- [ ] **Paso 3:** Commit de `requirements.txt`, `tests/conftest.py` y `salidas/salida_pipeline.txt`.

### Tarea 1: Pliegues anclados al calendario

**Files:**
- Modify: `TP3/config.yaml` (sección `validacion`: agregar `ancla_pruebas` y `n_pliegues_eleccion`, sacar `n_pliegues`; agregar sección `experimentos` con las claves de Global Constraints)
- Modify: `TP3/src/validacion.py` (`pliegues_walk_forward`, `__main__`, docstring)
- Modify: `TP3/src/modelado.py` (`pliegues_eleccion_reporte`)
- Test: `TP3/tests/test_validacion.py`

**Interfaces:**
- Produces: `pliegues_walk_forward(dias: DatetimeIndex, hasta: Timestamp | None = None, ancla=None, dias_prueba=None, gap_dias=None, dias_min_entrenamiento=None) -> list[tuple[DatetimeIndex, DatetimeIndex]]`. Mantiene el nombre para no romper a los que la llaman.
- Produces: `dias_en_espera(dias, hasta=None) -> DatetimeIndex`: días usables posteriores al último bloque completo.
- Produces: `pliegues_eleccion_reporte(hasta=None) -> (list[(k, (tr, te))], list[(k, (tr, te))])`. Elección: los primeros `n_pliegues_eleccion`; reporte: el resto. `k` empieza en 1.

- [ ] **Paso 1: Tests que fallan**, sobre una lista sintética de días = 10/09…14/10 + 16/10, 17/10, 18/10, 21/10, 29/10 (los días usables que había el 16/11):
  - `test_pliegues_iguales_a_los_actuales`: los bloques de prueba 1–3 son 30/09–04/10, 05/10–09/10 y 10/10–14/10; el 4 es {16, 17, 18, 21, 29}/10.
  - `test_dia_nuevo_no_cambia_pliegues_previos`: agregar 02/11 deja los 4 pliegues idénticos (entrenamiento y prueba) y `dias_en_espera == [02/11]`.
  - `test_bloque_completo_agrega_pliegue`: agregar 02/11…06/11 genera el pliegue 5, con prueba 02–06/11 y entrenamiento hasta el 30/10 o antes.
  - `test_hasta_recorta`: con `hasta=2026-10-14`, devuelve 3 pliegues.
  - `test_verificar_pasa`: `verificar(pliegues, 2)` no lanza ninguna excepción en todos los casos anteriores.
- [ ] **Paso 2:** `.venv/bin/python -m pytest tests/test_validacion.py -v` → FAIL.
- [ ] **Paso 3: Implementar.** Tomar los días usables ≥ ancla y ≤ `hasta`, cortarlos en grupos de `dias_prueba` y descartar el último grupo si está incompleto. El entrenamiento de cada pliegue se arma igual que hoy (gap en días de calendario, mínimo de días). `__main__` imprime la tabla de pliegues con una columna "uso" (elige / reporta) y la línea "Días en espera de completar un bloque: …". El docstring explica la regla del ancla.
- [ ] **Paso 4:** Tests → PASS. `.venv/bin/python src/validacion.py` con los datos reales muestra 5 pliegues (salvo que la calidad haya excluido días nuevos) y los días en espera.
- [ ] **Paso 5:** Commit.

### Tarea 2: Caché por pliegue con huella de contenido

**Files:**
- Create: `TP3/src/cache.py`
- Test: `TP3/tests/test_cache.py`

**Interfaces:**
- Produces: `huella_dias(dias: Iterable[Timestamp], carpeta: Path = FEAT_DIR) -> dict[str, str]`: sha256 del contenido de cada `data/features/<dia>.parquet` (`hashlib.file_digest`).
- Produces: `clave(**partes) -> str`: sha256 del JSON ordenado de `partes` (días, huellas, columnas, hiperparámetros, versión del modelo).
- Produces: `leer(tipo: str, clave: str) -> object | None` y `guardar(tipo: str, clave: str, objeto) -> None`: pickle en `data/cache/<tipo>/<clave>.pkl`.

- [ ] **Paso 1: Tests que fallan** (con `tmp_path` y parquet chicos):
  - `test_misma_entrada_misma_clave`.
  - `test_cambia_contenido_cambia_clave`: se reescribe un día con un valor distinto y la clave cambia.
  - `test_reescribir_igual_no_cambia_clave`: se reescribe el mismo DataFrame y se hace `os.utime` con otra fecha; la clave no cambia (Review Focus 3). Este test tiene que **reescribir el parquet con pandas**, para comprobar que la escritura es determinista.
  - `test_cambia_lista_de_dias_cambia_clave` (Review Focus 2).
  - `test_guardar_leer` y `test_leer_inexistente_devuelve_none`.
- [ ] **Paso 2:** FAIL. **Paso 3:** implementar. **Paso 4:** PASS.
- [ ] **Paso 5:** Commit.

### Tarea 3: Ridge para cualquier lista de familias (producto sin flags)

**Files:**
- Modify: `TP3/src/modelo_interpretable.py` (completo: sacar los globales `EXP_*`, `REZAGOS`, `POR_HORIZONTE` y el modo `--todas`)
- Modify: `TP3/src/modelado.py` (sacar `EXP_*` y `SUFIJO`)
- Modify: `TP3/config.yaml` (`modelos.interpretable.familias`: inicialmente las 8; `ridge_alphas`: la grilla que hoy es `ridge_alphas_todas`; `max_filas`: la que hoy es `max_filas_todas`)
- Test: `TP3/tests/test_ridge_familias.py`

**Interfaces:**
- Consumes: `pliegues_eleccion_reporte` (T1), `cache.*` (T2).
- Produces: `TODAS = list(FAMILIAS)` (las 8, en el orden de `familias.json`).
- Produces: `columnas_de(familias: list[str]) -> list[str]`: columnas de `data/features` que hay que leer.
- Produces: `class Disenio(familias: list[str])` con `.fit(df)`, `.transform(df) -> np.ndarray[float32]`, `.nombres: list[str]` y `.familia_de: list[str]` (familia de cada columna, en el mismo orden).
- Produces: `ajustar(A: DataFrame, familias: list[str], alpha: float) -> (Disenio, Ridge)`.
- Produces: `contribucion_por_familia(dis, coef: np.ndarray, X: np.ndarray) -> pd.Series`: desvío estándar de Σβx por familia.
- Produces: `hiperparametros_vigentes(ruta: Path, familias: list[str]) -> dict | None`: `None` si no existe el archivo, si no tiene la clave `familias` o si las familias no coinciden.
- Columnas por familia (las fija el spec §8): spline de hora solar, `cond_*`, `enfriamiento_nocturno`, `prob_lluvia`, `horizonte` y las curvas de `temp_anom`, `dep_rocio` y `viento_vel` → `base`; las demás columnas de `base` (lineales con indicador de faltante, como en `--todas`) → `base`; curvas de `pendiente_pron` → `pendiente`; `err_*` de rezagos, `err_ult_x_h_s*` y `err_ayer_x_<cond>` → `rezagos`; los 4 `sesgo_ewma_*` y `sesgo_ewma_*_x_noche` → `sesgo_ewma`; `vec_*` y `vec_err_1h_x_h_s*` → `vecinos`; las columnas de `revision`, `rezagos_extra` y `dinamica` → su familia. Cada `falta_<c>` va con la familia de `<c>`.

- [ ] **Paso 1: Tests que fallan** (DataFrame sintético con todas las columnas de `familias.json`, unas 2.000 filas al azar):
  - `test_todas_las_familias_tienen_columnas`: cada una de las 8 aparece en `familia_de`.
  - `test_sacar_vecinos`: sin `vecinos` no hay ninguna columna que empiece con `vec_`.
  - `test_sacar_base_mantiene_ponderacion_por_horizonte` (Review Focus 5): sin `base` existen `err_ult_x_h_s*` y no existen `cond_*`, `hora_s*` ni `horizonte`.
  - `test_contribucion_suma_prediccion`: la suma por fila de las contribuciones por familia (antes de tomar el desvío) más el intercepto es igual a `modelo.predict` (con tolerancia de 1e-4).
  - `test_hiperparametros_viejos_no_sirven` (Review Focus 4): un JSON con `{"sesgo": "sesgo_ewma_10", "ridge_alpha": 1.0}` devuelve `None`.
- [ ] **Paso 2:** FAIL. **Paso 3:** implementar, reutilizando el armado actual de `_crudo` del modo `--todas`. **Paso 4:** PASS.
- [ ] **Paso 5: `main()` del producto.** Familias tomadas de `config.yaml`. Si `hiperparametros_vigentes` devuelve `None` (o se pasa `--reelegir`), se elige α en los pliegues de elección. Si no, se usa el α guardado. Cada pliegue de reporte se lee de la caché (`tipo="ridge"`) o se entrena y se guarda. Las salidas mantienen nombre y formato (`interpretable_por_pliegue.csv`, `interpretable_coeficientes.csv`, `interpretable_hiperparametros.json`, este último con las claves `familias` y `ridge_alpha`). La salida en pantalla sigue el estilo del pipeline.
- [ ] **Paso 6:** Commit.

### Tarea 4: Resolver muchos Ridge desde XᵀX

**Files:**
- Create: `TP3/src/ridge_gram.py`
- Test: `TP3/tests/test_ridge_gram.py`

**Interfaces:**
- Produces: `class Gram` con `desde(X: np.ndarray, y: np.ndarray, bloque=200_000) -> Gram`, que acumula XᵀX y Xᵀy en float64 por bloques; X ya viene estandarizada (centrada).
- Produces: `Gram.resolver(mascara: np.ndarray[bool], alpha: float) -> (coef: np.ndarray, intercepto: float)`, con coeficientes en cero donde la máscara es False. Intercepto = media de y.
- Por qué funciona: el escalado y los splines se ajustan columna por columna, así que sacar una familia equivale a sacar sus filas y columnas de XᵀX. El resultado es idéntico a reentrenar.

- [ ] **Paso 1: Tests que fallan:**
  - `test_igual_a_sklearn`: X aleatoria de 5.000×30, estandarizada con `StandardScaler`, y α ∈ {1, 1e3, 1e5}. Los coeficientes coinciden con `Ridge(alpha).fit(X, y).coef_` con `atol=1e-6`.
  - `test_mascara_igual_a_reentrenar_sin_columnas`: con 10 columnas sacadas, coincide con sklearn entrenado solo con las columnas restantes.
- [ ] **Paso 2:** FAIL. **Paso 3:** implementar con `np.linalg.solve`. **Paso 4:** PASS.
- [ ] **Paso 5:** Commit.

### Tarea 5: LightGBM con las 8 familias, hiperparámetros congelados y caché

**Files:**
- Modify: `TP3/src/modelo_lgbm.py`

**Interfaces:**
- Consumes: T1, T2. Produces: `X_COLS` = todas las columnas de `familias.json`; `matriz(df)` y `entrenar(A, params, n)` sin cambios de firma.

- [ ] **Paso 1:** Sacar el diccionario `EXPERIMENTOS` y `SUFIJO`. La grilla se corre solo si `lgbm_hiperparametros.json` no existe, no tiene la clave `familias` con las 8 familias, o se pasa `--reelegir`. Cada pliegue de reporte se lee de la caché (`tipo="lgbm"`, que guarda las filas de `evaluar`, la importancia y el modelo como texto) o se entrena y se guarda.
- [ ] **Paso 2: Verificación.** Correr `src/modelo_lgbm.py` dos veces. La segunda tiene que tardar menos de 1 minuto y mostrar en pantalla "pliegue k: leído de la caché". Ese tiempo queda anotado en el mensaje del commit.
- [ ] **Paso 3:** Commit.

### Tarea 6: Comparación y scripts que usan los resultados

**Files:**
- Modify: `TP3/src/comparar_modelos.py`, `TP3/src/interpretar.py`, `TP3/src/preparar_dashboard.py`, `TP3/src/diagnostico_viento.py`

**Interfaces:**
- Consumes: `MI.ajustar(A, familias, alpha)`, `MI.columnas_de(familias)`, `hp_lin["familias"]`, `hp_lin["ridge_alpha"]` (T3); `ML.COLUMNAS`, `ML.entrenar`, `ML.matriz` (T5).

- [ ] **Paso 1:** Reemplazar `MI.ajustar(A, hp_lin["sesgo"], …)` por `MI.ajustar(A, hp_lin["familias"], hp_lin["ridge_alpha"])` y `MI.COLUMNAS` por `MI.columnas_de(hp_lin["familias"])`. Sacar `SUFIJO`, `EXP_TODAS` y los flags. `comparar_modelos.py` guarda en caché las predicciones por día de cada pliegue (`tipo="comparacion"`).
- [ ] **Paso 2: Verificación.** `.venv/bin/python -c "import comparar_modelos, interpretar, preparar_dashboard, diagnostico_viento"` (desde `src/`) termina sin errores, y `grep -rn "SUFIJO\|EXP_" src/` no devuelve nada.
- [ ] **Paso 3:** Commit.

### Tarea 7: Lógica de la experimentación

**Files:**
- Create: `TP3/src/experimentos.py`
- Test: `TP3/tests/test_experimentos.py`

**Interfaces:**
- Consumes: T1 (`pliegues_eleccion_reporte(hasta=fecha_corte)`), T2, T3 (`Disenio`, `TODAS`, `contribucion_por_familia`), T4 (`Gram`).
- Produces:
  - `dias_exploracion() -> DatetimeIndex`: días usables ≤ último día de prueba del último pliegue de elección (con el corte). Con los datos actuales termina el 09/10.
  - `muestra_exploracion(n=500_000) -> DataFrame`: filas reales de esos días, con todas las columnas y semilla fija.
  - `correlaciones(df) -> DataFrame`: por variable, su familia, la correlación de Spearman con `error_c` en total y en los horizontes 1, 6, 24 y 48.
  - `redundancia_entre_familias(df) -> DataFrame`: matriz 8×8 con el R² de predecir la contribución de cada familia a partir de las otras. Para el mapa de calor.
  - `preparar_gram(k, tr, te) -> (Gram, X_prueba, T_prueba, Disenio)`, con caché `tipo="gram"`: un `Disenio(TODAS)` por pliegue de elección.
  - `evaluar_variante(familias, alphas) -> DataFrame`: por α y pliegue, el MAE de historia completa. Elige el α de menor MAE medio y devuelve el MAE por pliegue con ese α.
  - `contribuciones_todas() -> DataFrame`: una fila por familia × pliegue de elección, con la contribución (`contribucion_por_familia` sobre X de prueba, con el α elegido para todas) y la fracción de coeficientes que cambian de signo entre pliegues.
  - `ablacion() -> DataFrame`: una fila por familia, con `mae_sin_por_pliegue`, `delta_promedio`, `delta_max` (contra la variante con todas) y `se_saca` (según la regla).
  - `aplicar_regla(delta_promedio: float, delta_max: float, umbral_prom=0.005, umbral_pliegue=0.010) -> bool`: `delta_promedio <= umbral_prom and delta_max <= umbral_pliegue`.
  - `confirmar_recorte(a_sacar: list[str], costo: dict[str, float], evaluar) -> list[str]`: evalúa "todas menos `a_sacar`". Mientras empeore más de 0,005 °C contra todas, devuelve a la lista la familia con mayor `delta_promedio` y repite. Recibe `evaluar` por parámetro para poder testearla.
  - `exportar(nombre: str, df: DataFrame) -> Path`: escribe en `TP3/salidas/experimentos/<nombre>.csv`.
- Si el α elegido queda en un borde de la grilla, se agrega un orden de magnitud hacia ese lado y se repite (como máximo dos veces).

- [ ] **Paso 1: Tests que fallan:**
  - `test_regla`: (0,004; 0,009) → True; (0,006; 0,006) → False; (0,003; 0,012) → False; (−0,010; −0,002) → True; (0,005; 0,010) → True (los bordes cuentan).
  - `test_confirmar_recorte_devuelve_la_mas_costosa`: un `evaluar` falso empeora 0,008 si se sacan A y B juntas y 0,001 con cualquiera sola. Con costos A = 0,004 y B = 0,002, el resultado es `["B"]`.
  - `test_dias_exploracion_no_toca_reporte`: con días sintéticos, el máximo de `dias_exploracion()` es menor que el primer día de prueba del primer pliegue de reporte.
- [ ] **Paso 2:** FAIL. **Paso 3:** implementar. **Paso 4:** PASS.
- [ ] **Paso 5:** Commit.

### Tarea 8: Notebook `experimentos.ipynb`

**Files:**
- Create: `TP3/notebooks/experimentos.ipynb`, armada con `nbformat` desde un script descartable
- Create: `TP3/salidas/experimentos/*.csv` y `TP3/salidas/experimentos/figuras/*.png` (se generan al correr la notebook)

- [ ] **Paso 1: Estructura.**
  1. Introducción: qué decide la notebook, el corte, los pliegues de elección y por qué se explora solo hasta el 09/10.
  2. Regla de recorte con sus razones, copiadas del spec §4.
  3. Una sección por familia 1–8, con la **hipótesis experta** (tomada de DECISIONES.md y del encabezado de `features.py`), las **correlaciones con el error** (tabla y barras por horizonte) y un **boxplot** del error por tramos de la variable principal de la familia.
  4. Mapa de calor de redundancia entre familias.
  5. Ridge con todas: α elegido, MAE por pliegue y contribución y estabilidad por familia (barras con un punto por pliegue).
  6. Ablación: tabla con los deltas y la decisión de la regla, y el recorte confirmado.
  7. Celda `APARTARSE_DE_LA_REGLA = {}` (familia → razón escrita), vacía por defecto, que se aplica sobre el resultado.
  8. Resultado: la lista de familias del Ridge final y la tabla de variantes (MAE total, por horizonte y por segmento).
- [ ] **Paso 2:** `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/experimentos.ipynb` termina sin errores. Verificar que existen `salidas/experimentos/ablacion.csv` y `salidas/experimentos/familias_ridge_final.csv`.
- [ ] **Paso 3:** Commit de la notebook ejecutada y de `salidas/experimentos/`. **Mostrar el resultado al usuario antes de seguir**: el grupo puede querer apartarse de la regla.

### Tarea 9: Modelos finales, comparación y documentación

**Files:**
- Modify: `TP3/config.yaml` (`modelos.interpretable.familias` = el resultado de la T8)
- Modify: `TP3/notebooks/comparacion_modelos.ipynb` (solo agregar celdas: una nota arriba de todo que dice que ahora los pliegues son anclados y los de reporte son 3–5; después, re-ejecutar)
- Modify: `TP3/DECISIONES.md`, `TP3/README.md`

- [ ] **Paso 1:** Correr `modelo_interpretable.py`, `modelo_lgbm.py`, `comparar_modelos.py`, `interpretar.py` y `preparar_dashboard.py`, en ese orden. Guardar las salidas en pantalla en `salidas/modelos_finales.txt`.
- [ ] **Paso 2:** Re-ejecutar `comparacion_modelos.ipynb` con nbconvert.
- [ ] **Paso 3:** DECISIONES.md: resultado de la experimentación (familias del Ridge final), resultados finales contra B0–B3 en los pliegues 3–5 y la diferencia entre LightGBM y Ridge (día como unidad). README: comandos nuevos (`--reelegir`, notebook de experimentos, `pytest`) y la regla de los pliegues anclados.
- [ ] **Paso 4:** `.venv/bin/python -m pytest -q` → todos pasan. Correr `src/pipeline.py` una vez más: la salida no cambia y la caché no se invalida.
- [ ] **Paso 5:** Commit. Pedirle al usuario que revise los resultados finales.
