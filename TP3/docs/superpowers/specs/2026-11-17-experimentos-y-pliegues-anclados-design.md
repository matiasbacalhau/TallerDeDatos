# Diseño — Experimentación por familias y pliegues anclados al calendario

Fecha: 17/11/2026 · Rama: `tp3-experimentos` · Estado: borrador para aprobar

## 1. Intención

El trabajo final del TP3 es **el modelo** (producto). Antes de entrenarlo hay que **justificar qué variables usa el
modelo de caja de cristal**, con un recorrido que se pueda leer como documento: hipótesis experta → exploración
visual → experimentos. Hoy los experimentos están dispersos (flags `--vecinos`, `--revision`, `--rezagos`,
`--dinamica`, `--todas`; decenas de CSV con sufijo `_exp_*`; conclusiones escritas a mano en DECISIONES.md) y
corrieron con 47 días. Ahora hay 55 (10/09 → 09/11).

La pregunta que el producto responde: **¿un modelo de caja negra con toda la información (LightGBM) es
sustancialmente mejor que uno regularizado y de caja de cristal (Ridge)?** Los dos se comparan contra el NWS crudo
(B0) y los baselines (B1–B3).

## 2. Orden del trabajo

1. **Datos:** correr `src/pipeline.py` para incorporar los 8 días nuevos (02/11 → 09/11).
2. **Parte A — Pliegues anclados** (producto preparado para crecer).
3. **Parte B — Experimentación** con fecha de corte congelada: decide las familias del Ridge.
4. **Modelos finales:** Ridge con las familias elegidas y LightGBM con las 8 familias, sobre los pliegues anclados,
   comparados contra B0–B3.

## 3. Parte A — Pliegues anclados al calendario (opción A)

**Objetivo:** que un día nuevo no obligue a recalcular lo ya calculado. Aunque el TP no vaya a sumar más datos, el
producto tiene que quedar listo para hacerlo.

**Regla nueva (reemplaza «los últimos 4 pliegues»):**
- Los bloques de prueba se definen desde una **fecha ancla fija** (`validacion.ancla_pruebas` en `config.yaml`):
  bloques consecutivos de `dias_prueba` (5) **días usables** (no excluidos por calidad), en orden, desde la ancla.
  Igual que hoy, un bloque puede saltear días excluidos o sin archivo.
- Un bloque existe solo cuando tiene sus 5 días. Los días sobrantes al final esperan a completar el siguiente.
- Entrenamiento: todos los días usables hasta `gap_dias` (2) días de calendario antes del primer día de prueba
  (igual que hoy). Mínimo `dias_min_entrenamiento` (10).
- **Elección / reporte:** los primeros `n_pliegues_eleccion` (2) pliegues **eligen los hiperparámetros una sola
  vez**. Quedan congelados en `*_hiperparametros.json`. Los pliegues siguientes **solo se evalúan** (reporte y
  monitoreo). Se vuelve a elegir solo a pedido (`--reelegir`).
- Ancla propuesta: **30/09/2026**, que coincide con el primer bloque de prueba actual. Así, con los datos que ya se
  usaban, los pliegues 1–3 quedan iguales a los de hoy.

**Caché por pliegue:** cada resultado (modelo × pliegue) se guarda con una **huella** de lo que lo determina: días
de entrenamiento y de prueba, huella de los datos de esos días (`manifiesto_pares.json` o equivalente sobre
`data/features/`), columnas usadas, hiperparámetros y configuración del modelo. Al correr, si la huella coincide, se
reutiliza y no se entrena. Un día nuevo solo genera trabajo cuando completa un bloque, o si cambió un día ya usado
(por ejemplo, cuando un hueco interpolado se cierra).

**Archivos afectados:** `validacion.py` (regla), `modelado.py` (`pliegues_eleccion_reporte`, caché),
`modelo_interpretable.py`, `modelo_lgbm.py`, `comparar_modelos.py`, `baselines.py` (es barato: puede seguir
recalculando todo, pero sobre los mismos pliegues). Usuarios de los resultados que hay que revisar:
`interpretar.py`, `preparar_dashboard.py`, `diagnostico_viento.py`, `notebooks/comparacion_modelos.ipynb`,
`app/dashboard.py`. Los CSV mantienen su formato (columna `pliegue`) para no romper a esos usuarios.

**Controles:** `verificar()` sigue valiendo (sin solapes, con gap, bloques consecutivos). Prueba nueva: agregar un
día simulado no cambia los pliegues anteriores ni sus resultados guardados.

## 4. Parte B — Experimentación (opción B, corte congelado)

**Corte:** `experimentos.fecha_corte: 2026-11-09` en `config.yaml`. Los experimentos usan los pliegues anclados con
días ≤ corte. Los días posteriores no reabren nada. Mover el corte es una decisión explícita del grupo.

**Contra qué se decide:** solo los **pliegues de elección**. La exploración usa solo días hasta el final del último
bloque de prueba de elección. Ningún día de prueba de los pliegues de reporte interviene en la decisión: si
interviniera, el resultado final quedaría optimista (es leakage en la decisión del analista).

**Formato elegido: una sección por familia (1 a 8).** Después de verlo, el grupo decide si conviene más el formato
acumulativo («sumar familias»), según cuál quede más claro y didáctico.

**Entregable:** `notebooks/experimentos.ipynb`, que **corre** los experimentos (con caché: si ya están calculados
con la misma huella, los lee) y **exporta** los resultados. Las cuentas pesadas viven en `src/experimentos.py` (la
notebook llama funciones, no reimplementa modelos). La notebook se commitea ejecutada, con sus gráficos y tablas.

**Pasos de la notebook:**

1. **Hipótesis experta por familia (1–8):** por qué podría corregir el error del NWS. La base es el mecanismo
   meteorológico (ya escrito en DECISIONES.md y en el encabezado de `features.py`).
2. **Exploración por familia** (días de exploración, observaciones reales):
   - correlación de cada variable con el error del NWS (`error_c`), en general y por horizonte;
   - mapa de correlaciones **entre familias** (redundancia: si dos dicen lo mismo, una sobra);
   - boxplots del error por categorías relevantes (condición, noche, tramos de horizonte, tramos de la variable).
3. **Ridge con todas las familias:** la especificación de `--todas` actual (todas las columnas lineales con
   indicador de faltante, las curvas, las interacciones y la penalización α elegida en una grilla), en los pliegues
   de elección. Se reporta:
   - **contribución por familia**: desvío estándar de la parte de la predicción que aporta cada familia
     (Σ β·x de sus columnas). Se juzga por familia y no columna por columna: las columnas parecidas se reparten el
     peso, y una curva se juzga entera;
   - **estabilidad** de esa contribución y de los signos entre los pliegues de elección.
4. **Ridge sin cada familia (ablación) y regla de recorte fijada de antemano** (acordada el 17/11/2026, antes de
   ver resultados). Por cada familia se entrena un Ridge **sin ella**, con su propia búsqueda de α, en los pliegues
   de elección. La familia **se saca** si se cumplen las dos condiciones:
   1. sin ella, el MAE promedio de los pliegues de elección no empeora en más de **0,005 °C** respecto del Ridge con
      todas las familias;
   2. en ningún pliegue de elección empeora en más de **0,010 °C**.

   El Ridge final es «todas menos las que se sacan» y se reentrena con su búsqueda de α. Se confirma que su MAE en
   los pliegues de elección no empeora en más de 0,005 °C respecto del Ridge con todas. Si empeora (dos familias
   que, cada una por separado, no hacían falta, pero sí juntas), se devuelve la de menor costo de sacarla y se
   repite. La contribución y la estabilidad del paso 3 **explican** por qué una familia se va o se queda, pero no
   deciden. El grupo puede apartarse de la regla solo dejando escrita la razón en la notebook.

   **Por qué estos umbrales:**
   - **Fijados antes de ver resultados (pre-registro).** Si el umbral se elige mirando las tablas, sin querer se
     termina eligiendo el que favorece el resultado esperado. Escrita antes, la regla se puede defender y la puede
     repetir un tercero. Además se aplica sola cuando lleguen datos nuevos (opción A).
   - **0,005 °C es «no distinguible».** Equivale a menos del 0,5 % del MAE (≈ 1,3 °C). Está por debajo del ruido
     entre días que ya medimos: con `--todas`, LightGBM mejoró 0,012 °C por día y no fue significativo
     (p = 0,12; IC95 −0,026 a 0,000; DECISIONES.md, familia 8). Una familia cuya salida cambia menos que eso no
     aporta nada que podamos detectar con estos datos, y suma columnas que hacen al modelo menos legible.
   - **0,010 °C por pliegue** evita sacar una familia que en promedio no importa, pero que en un período concreto
     (por ejemplo, una semana con frentes) sí ayuda. Es el doble del umbral promedio: tolera el ruido de un solo
     pliegue sin tapar un empeoramiento real.
   - **El daño de elegir mal está acotado:** la decisión usa solo los pliegues de elección, y el error del informe
     se mide en los de reporte, que no intervienen.
   - **Costo:** 8 corridas extra del Ridge (≈ 80 min de cómputo, según la corrida de `--todas` del 16/11).
5. **Resultado:** tabla con cada variante (MAE total y por horizonte, por segmento de historia) y la **lista de
   familias del Ridge final**. El Ridge más eficiente sale de este recorrido (regularización + recorte por la regla);
   el Ridge actual del producto (familias 1–4 elegidas a mano) no se usa como referencia.

**Exportación:** tablas chicas (CSV) en `TP3/salidas/experimentos/` (versionadas) y figuras en `TP3/figuras/`
(o `salidas/experimentos/figuras/`). Los intermedios pesados y la caché van en `data/experimentos/` (no versionado).

**LightGBM no participa en la selección:** se entrena siempre con las 8 familias.

## 5. Modelos finales

- **Ridge:** la especificación del paso 3 restringida a las familias elegidas. Hiperparámetros elegidos en los
  pliegues de elección.
- **LightGBM:** las 8 familias, con la misma grilla de hoy.
- Evaluación en los pliegues de reporte anclados, contra B0–B3, con `comparar_modelos.py` (día como unidad).
- Los resultados finales reemplazan a los del producto actual, que tenía las familias 1–4 en los dos modelos.

## 6. Fuera de alcance

- Nuevas familias de variables o cambios en `features.py`.
- Cambiar de modelo de caja de cristal (Elastic Net ya se probó y se descartó; ver DECISIONES.md).
- Rediseñar el dashboard (solo se adapta si cambian nombres de archivos).

## 7. Restricciones

- Reproducible con `make`/un comando por parte. Sin rutas ni parámetros hardcodeados (todo en `config.yaml`).
- No borrar celdas del grupo en `comparacion_modelos.ipynb`: lo nuevo va en celdas o archivos aparte.
- DECISIONES.md registra solo acuerdos (reglas nuevas: pliegues anclados, corte de experimentos, selección del Ridge).
- Salida en pantalla legible como informe, con el mismo criterio que el pipeline (`src/formato.py`).
- Plazo: TP3 vence el 18/11/2026 (campus hasta el 19/11 23:59).

## 8. Decisiones que quedan al implementador

- **Familia de las columnas derivadas** (decisión del implementador): spline de hora solar, condición y
  enfriamiento nocturno → base; curvas de `pendiente_pron` → pendiente; `err_ult × horizonte` y `err_ayer × condición`
  → rezagos; sesgo exponencial × noche → sesgo EWMA; `vec_err_1h × horizonte` → vecinos. Los indicadores de faltante
  van con su variable.
- Formato exacto de la huella y del archivo de caché.
- Grilla de α del Ridge (partir de `ridge_alphas_todas`). Si el α elegido queda en un borde de la grilla, se
  amplía la grilla.
- Tipo y cantidad de gráficos de la exploración, mientras cubran los tres de la sección 4.
