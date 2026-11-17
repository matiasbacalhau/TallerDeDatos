# TP3 — Registro de decisiones (rama `tp3-experimentos`)

Se actualiza a medida que el grupo decide. Sirve de base para el informe y la defensa.

En esta rama el TP3 se corrió **desde cero** con 55 días (10/09 → 09/11/2026). Quedan las decisiones tomadas en `main` que siguen valiendo acá; los resultados y las decisiones que solo valían para el trabajo hecho en `main` (35 y 47 días, 4 pliegues, Ridge con variables elegidas a mano, experimentos con banderas) no están. Todos los números son de esta rama.

## Paso 0 — Datos (2/11/2026)

| # | Decisión | Elegido | Por qué / evidencia |
|---|---|---|---|
| 1 | Dataset | El del grupo: CSV diarios de `../archivos/` (55 días, 10/09 → 09/11) | La consigna pide el «dataset fusionado del TP1». |
| 2 | Variable objetivo | Temperatura, `error_c = observado − pronosticado` | Es la variable del TP2. |
| 3 | Horizontes | Los 48 (1–48 h) | Decisión del grupo; 32,6 M de filas. Para entrenar se submuestrea. |
| 4 | Observación faltante | Interpolación lineal en el tiempo si el hueco dura ≤ 3 h; cada valor interpolado queda marcado (`obs_interpolada`) | Prueba de ocultamiento sobre lecturas reales (hecha con 35 días): error absoluto medio 0,43 °C (hueco de 1 h), 0,55 (2 h), 0,71 (3 h). Con 55 días: 62,1 % de las estación-hora con lectura real, 11,0 % interpoladas, 26,9 % sin dato. |
| 5 | Evaluación | Baselines: se evalúan con filas reales e interpoladas. Modelos: se entrenan con ambas y se evalúan **solo en filas reales**. La tabla final compara todo sobre filas reales | Las métricas no dependen de valores estimados. |

Resultado: `data/pares/*.parquet` (formato largo: una fila por estación × hora objetivo × horizonte), generado con `src/preparar_datos.py`.

## Etapa A — Variables

- **Restricción de causalidad (lógica, no negociable):** un error rezagado solo puede usarse si se observó a una distancia ≥ horizonte respecto de la hora objetivo, es decir, hasta la emisión (`forecast_generated_at_utc`). Lo controla una prueba automática en cada corrida.
- **Ventanas de rezagos elegidas con los datos.** Autocorrelación del error del pronóstico a 24 h (por estación, sin el sesgo medio, solo observaciones reales; medida con 35 días, 840 horas por estación):

  | Distancia | 1 h | 2 h | 3 h | 6 h | 9 h | 12 h | 18 h | 24 h | 36 h | 48 h |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Autocorr. | 0,84 | 0,69 | 0,54 | 0,19 | −0,03 | −0,10 | 0,06 | 0,35 | −0,15 | 0,24 |

  Persistencia real de unas 6 h; los picos a 24 y 48 h son una interpretación (ciclo diario del error), no comprobada. Cada familia se justifica además con la ablación de la experimentación.
- **Sesgo persistente de la estación:** promedio exponencial de días (como el decaying average del NBM del NWS) con 4 memorias (α = 0,05, 0,1, 0,2 y 0,3). Los dos modelos reciben los 4; el Ridge los pondera con su penalización.

## Etapa A — Validación

- **Walk-forward (expanding window) con gap**, sin bloque final reservado (decisión del grupo, 2/11; así todo el período se usa para elegir y no queda una prueba limpia).
- Cortes por **día de la hora objetivo**, nunca por filas sueltas: cada hora objetivo aparece hasta 48 veces (una por horizonte) con la misma observación.
- **Pliegues anclados al calendario (17/11):** los bloques de prueba se cuentan desde el 30/09/2026, de a 5 días usables (no excluidos). Un día nuevo no cambia los pliegues anteriores: aparece un pliegue nuevo cuando se completa un bloque. Gap de 2 días de calendario y mínimo de 10 días de entrenamiento (`config.yaml` → `validacion:`).
- Con los 55 días: 5 pliegues. Los pliegues 1–2 eligen hiperparámetros (pruebas 30/09–04/10 y 05/10–09/10) y los 3–5 reportan (10/10–14/10; 16, 17, 18, 21 y 29/10; 02/11–06/11). Los días 07–09/11 esperan a completar un bloque.
- Código: `src/validacion.py` (`pliegues_walk_forward`, `dias_en_espera`, `verificar`).
- **No** se agrega validación dejando estaciones afuera (por ahora).

## Datos que crecen

El grupo actualiza `archivos/` periódicamente.
- `src/preparar_datos.py` es **incremental**: procesa solo los CSV nuevos y reescribe un día de `pares/` únicamente si cambió su observación (p. ej. el día anterior, cuando un hueco se cierra con datos del día nuevo). `--rehacer` fuerza todo de cero.
- **Los modelos también son incrementales** (17/11): cada resultado (modelo × pliegue) queda en una caché con la huella del contenido de sus datos (`src/cache.py`). Control: con el pipeline vuelto a correr completo, los dos modelos leyeron sus 3 pliegues de reporte de la caché (3 s en lugar de 2–24 min).
- **Fecha de corte de los experimentos:** 09/11/2026 (`config.yaml` → `experimentos.fecha_corte`).

## Etapa A — Baselines (2/11/2026)

- **Hora local = hora solar** (hora UTC + longitud/15, redondeada). No tiene el salto del cambio de hora civil del 1/11/2026 y sigue el ciclo del sol. El dashboard también muestra la hora solar (ver Etapa C).
- **Baselines (la consigna dice «modelos baseline», en plural):** B0 NWS crudo (c = 0), B1 sesgo global, B2 sesgo por estación × hora solar, B3 persistencia del último error conocido. Un tramo por horizonte en B2 no mejoró (1,398 contra 1,396, medido con 35 días) y se descartó.
- **Aprenden solo con observaciones reales del entrenamiento.** Se evalúan con filas reales y con reales + interpoladas, siempre sobre las **mismas filas** (donde B2 y B3 están definidos).
- **Causalidad de B3:** la observación de la hora H puede llegar hasta H + 59 min y el pronóstico se genera entre 60 min antes y 20 min después de «objetivo − horizonte». Por eso el último error usable es el de la hora objetivo − horizonte − 2. Una primera versión usaba objetivo − horizonte y daba 0,703 °C a 1 h: era leakage (el valor correcto es 1,24). `prueba_causalidad` lo controla en cada corrida (20,9 M de filas, 0 violaciones).
- Código: `src/baselines.py`. Resultados: `data/resultados/baselines_por_pliegue.csv` y `baselines_resumen.csv`.

MAE medio de los 5 pliegues (°C, filas reales):

| Horizonte | B0 NWS crudo | B1 global | B2 estación×hora | B3 persistencia |
|---|---|---|---|---|
| todos | 1,487 | 1,488 | **1,393** | 1,962 |
| 1 h | 1,360 | 1,362 | **1,277** | **1,277** |
| 3 h | 1,381 | 1,382 | **1,293** | 1,609 |
| 6 h | 1,398 | 1,399 | **1,303** | 1,900 |
| 12 h | 1,405 | 1,408 | **1,332** | 2,042 |
| 24 h | 1,493 | 1,494 | **1,392** | 1,765 |
| 48 h | 1,636 | 1,634 | **1,536** | 2,024 |

RMSE (todos): 1,877 / 1,880 / 1,763 / 2,516. Sesgo residual: +0,078 / +0,005 / +0,034 / +0,034.

Lecturas:
- B2 mejora a B0 un 6,3 % en MAE: hay **sesgo condicional** por estación y hora aprendible.
- B1 no mejora a B0: el sesgo global no es estable entre semanas.
- B3 solo empata o gana a 1 h y pierde desde 3 h.

## Etapa A — Feature engineering

Familias de variables (`src/features.py`, que genera `data/features/<dia>.parquet` y `familias.json`; control en `src/verificar_features.py`). Las 8 se calculan para todos los días; qué familias usa el Ridge lo decide la experimentación, y LightGBM usa las 8. Quedan fuera: mes, día del año, día de la semana (con 2–3 meses no se puede aprender un ciclo anual y el mes funcionaría como identificador del período) e identificador de la estación como variable (memorizaría; su sesgo habitual lo capta la familia 4).

| Familia | Variables |
|---|---|
| 1 base | `temp_pron`, `temp_anom`, `elev_solar` y `noche`, `hum_pron`, `rocio_pron`, `dep_rocio`, `viento_vel`, `viento_sin`, `viento_cos`, `viento_calmo`, `prob_lluvia`, `condicion` (grupos por palabras clave), `horizonte`, `hora_solar_sin/cos`, `lat`, `lon` |
| 2 pendiente | `pendiente_pron` = (pronóstico de T+1 con h+1 − pronóstico de T−1 con h−1) / 2, del **mismo ciclo de emisión** (verificado: 100 % de las filas coinciden en la hora de generación, diferencia ≤ 5 min) |
| 3 rezagos | `err_ult`, `err_media3`, `err_media6` (mismo horizonte, hasta T−h−2) y `err_ayer` (mismo horario, T−24·k, con 24·k ≥ h+2) |
| 4 sesgo_ewma | `sesgo_ewma_05/10/20/30`: promedio exponencial (α = 0,05/0,1/0,2/0,3) de la media diaria del error de la estación a esa hora, hasta el día D−k |
| 5 vecinos | `vec_err_ult` (último error de las vecinas, mismo horizonte) y `vec_err_1h` (error a 1 h de las vecinas). Vecinas: las 5 más cercanas dentro de 100 km (4,6 por estación en promedio; ninguna queda sin vecinas) |
| 6 revision | `rev_3h/6h/12h/24h`: pronóstico actual − pronóstico para la misma hora emitido 3, 6, 12 y 24 h antes |
| 7 rezagos_extra | `err_hace9h` … `err_hace21h` (mismo horizonte, 9–21 h antes de la emisión) y `err1h_ult`, `err1h_media2/4/8` (error del pronóstico a 1 h de la propia estación) |
| 8 dinamica | `obs_tend3h`, `err1h_tend3h`, `salto_obs` (tendencia y salto), `err_rocio_ult`, `err_rocio_1h` (error del punto de rocío), `err_std6`, `err_std24` (desvío del error), `dispersion_emis`, `amplitud_pron` |

- **Rocío observado (familia 8):** sale de `src/rocio_observado.py`, que lee los CSV del TP1 sin modificarlos y deja `data/obs_rocio.parquet`. Mismo criterio de validez que la temperatura (`match_type == "interval"`), solo valores reales.
- **Causalidad:** la fuente más reciente de cualquier variable es la hora objetivo − horizonte − 2; los pronósticos usados se emitieron en la emisión de la fila o antes. Control sobre las 32,6 M de filas: 0 violaciones.
- **Recálculo independiente** con pandas de 1.500 filas (25 estaciones; vecinas con otro método, BallTree): 0 discrepancias en las 34 variables derivadas.
- **Faltantes:** `err_ult` 35,8 %, `err_media3` 32,8 %, `err_media6` 28,5 %, `err_ayer` 37,3 %, `sesgo_ewma_*` 11,1 %, `temp_anom` 9,3 %, vecinos ~15 %, rezagos extra 26–38 %, dinámica 1–49 %. LightGBM los tolera; el Ridge imputa 0 con un indicador de faltante.
- **Viento:** dirección faltante (18,3 % de las filas) equivale a velocidad 0 km/h: se trata como calma (`viento_calmo`, sen/cos = 0).
- **Condición** (% de filas): despejado 50,0, lluvia 13,7, parcial 13,3, nublado 12,8, tormenta 6,1, niebla 3,0, otro 1,1, nieve 0,1. Tormenta incluye `T-storms`, huracán y tormenta tropical.
- **Asociación con el error:** ver la exploración de `notebooks/experimentos.ipynb` (correlaciones por familia y horizonte, solo días de exploración).

## Datos faltantes y falla de observaciones del NWS

- **Formato de los archivos:** los archivos que llegaron como `forecast_observation_v2_DDMMYYYY.csv` se **renombraron al formato único** `forecast_observation_YYYY_MM_DD.csv` (solo cambió el nombre). `preparar_datos.py` acepta los dos formatos.
- **Días sin archivo:** 15/10, 27/10, 28/10, 30/10, 31/10 y 01/11. El script los avisa y la validación usa días de calendario para el gap.
- **Falla de observaciones.** Fracción de filas sin observación emparejada (`unmatched`, normal 4–5 %):
  - 19/10: 18 % (2 horas con todas las estaciones sin observación); 20/10: 15 % (1 hora); 22/10: 44 % (10 horas con >90 %).
  - **23 al 26/10: 56–57 %.** Cada día, **272 de 519 estaciones no tienen ninguna observación** (contra 9–13 habituales), y son **las mismas 272 los cuatro días**. Están repartidas en 23 estados (FL 91 %, VT 83 %, GA 78 %… NY 25 %), así que no es un bloque geográfico ni alfabético.
  - **Causa confirmada (10/11):** fue una falla del NWS, no de nuestra Lambda ni de la fusión.
- **Decisión (10/11, acordada con el docente):** esos días salen del análisis. La exclusión **no es una lista manual**: la hace una regla automática (sección siguiente), que los detecta sola.

## Calidad de los datos: exclusión automática de días y alarmas (10/11/2026)

**Principio:** el producto recibe datos nuevos todo el tiempo; agregar días no puede romper el pipeline ni requerir tocar código. Por eso la decisión de excluir un día sale de una regla que se aplica a cada día que llega, y queda registrada con el dato que la justifica.

- **Regla (relativa):** un día se excluye si su % de filas sin observación supera en más de **10 pp** la **mediana de los 14 días válidos anteriores** (mínimo 7 días de referencia). La referencia usa solo días válidos, así una falla larga no se vuelve «normal».
- **Por qué relativa y no un umbral fijo (X %):** el nivel normal cambia. En septiembre hay ~25 % de filas sin observación, en octubre ~7 %. Un umbral del 15 % sacaría casi todo septiembre; uno del 30 % dejaría pasar el 19 y el 20/10.
- **Alarmas (solo avisan, no excluyen):**
  - Por día: cobertura total (real + interpolada) < 70 % en un día incluido. Detecta degradaciones lentas que la regla relativa iría absorbiendo. Se mide sobre la total y no sobre la real porque la real de septiembre es 64–70 % y la alarma sonaría siempre.
  - Del sistema: menos del 70 % de días usables en los últimos 14 días de calendario.
- **Resultado con los 55 días:** se excluyen **19, 20 y 22–26/10** (7 días; quedan 48). El 21/10 queda. Ningún día del 02–09/11 se excluye. Ningún día incluido dispara la alarma por día; la del sistema sí suena (9 de 14 días usables entre el 27/10 y el 09/11, por los días sin archivo).

  | Día | Sin observación | Referencia | Estaciones sin obs. |
  |---|---|---|---|
  | 19/10 | 21,7 % | 7,3 % | 13/519 |
  | 20/10 | 17,9 % | 7,3 % | 11/519 |
  | 22/10 | 48,1 % | 7,3 % | 18/519 (10 h de apagón) |
  | 23–26/10 | 55,9–58,3 % | 7,3 % | 281–282/519 |

- **Dónde queda documentado:** `data/calidad/REGISTRO_CALIDAD.md` (se regenera en cada corrida, con los avisos de cambios en la lista), `calidad_por_dia.csv` y `dias_excluidos.json`. Parámetros en `config.yaml` (`calidad:`).
- **Cómo se aplica:** `validacion.dias_disponibles()` quita los excluidos, así que baselines y modelos no los ven sin cambiar código. Las features se calculan igual para todos los días: las observaciones reales de un día excluido pueden servir de rezago a días posteriores.
- **Controles:** un día nuevo normal simulado queda incluido; uno con 60 % de faltantes queda excluido con aviso; un día con 65 % de cobertura que pasa la regla (contexto tipo septiembre) dispara la alarma. Una segunda corrida no reescribe nada.
- **Un solo comando:** `src/pipeline.py` corre preparación → calidad → validación → baselines → features → verificación, y se detiene si algún control falla. Su salida se lee como informe (`salidas/salida_pipeline.txt`).
- **Pendiente:** entender por qué septiembre tiene ~25 % de filas sin observación contra ~7 % en octubre.

## Días con historia incompleta y métricas por segmento (10/11/2026)

- **Regla:** un día válido (no excluido) tiene **historia incompleta** si falta o se excluyó alguno de los **3 días de calendario previos**. La marca **no excluye**: el target del día es bueno; lo que falta es parte de sus features con memoria (rezagos).
- **Por qué 3 días:** los rezagos llegan hasta 55 h atrás (h = 48, más 2 h de latencia, más la media de 6 h). Parámetro: `calidad.dias_historia`.
- **Para qué:** separar dos preguntas que el promedio mezcla. **Historia completa:** cuánto corrige el modelo con toda su información; es la **comparación entre modelos del informe**. **Historia incompleta:** cuánto corrige cuando le falta la historia reciente; mide la robustez ante caídas de la fuente, que en producción van a repetirse.
- **Dónde queda:** la lista de días marcados se genera en cada corrida en `data/calidad/REGISTRO_CALIDAD.md` y `historia_incompleta.json`. Con los 55 días: 11 días marcados (10–12/09, 16–18/10, 21/10, 29/10 y 02–04/11). Los resultados de baselines y modelos llevan la columna `segmento` (todas / completa / incompleta).

MAE de los baselines (°C, filas reales):

| Segmento | Pliegues | B0 NWS | B1 global | B2 estación×hora | B3 persistencia | B3 a 1 h | B2 a 1 h |
|---|---|---|---|---|---|---|---|
| completa | 1, 2, 3, 5 | 1,462 | 1,465 | **1,391** | 1,934 | **1,237** | 1,262 |
| incompleta | 4, 5 | 1,539 | 1,535 | **1,406** | 1,990 | 1,338 | **1,323** |
| todas | 1–5 | 1,487 | 1,488 | **1,393** | 1,962 | 1,277 | 1,277 |

Lecturas:
- B2 no usa rezagos, así que rinde parecido en los dos segmentos (1,391 y 1,406). La persistencia (B3) es la que más se degrada: a 1 h pasa de 1,237 a 1,338.
- **Límite:** B3 solo se mide donde su rezago existe (las mismas filas para todos), así que el segmento «incompleta» de los baselines subestima el problema: las filas sin rezago no entran. Los modelos se evalúan en todas las filas.

## Etapa A — Modelos

### Modelo de caja de cristal: Ridge con curvas

- **Regresión lineal regularizada (Ridge) con curvas (splines).** Cada variable aporta un coeficiente o una curva legible (scikit-learn: `SplineTransformer` + `Ridge`). Recibe todas las columnas de las familias que elige la experimentación (`config.yaml` → `modelos.interpretable.familias`).
- **Formas:**

  | Término | Forma | Mecanismo |
  |---|---|---|
  | Hora solar | curva periódica | Ciclo diario y transiciones de mañana y tarde |
  | `pendiente_pron` | curva | *Timing* de frentes y del ciclo diario; desfase de la observación |
  | `temp_anom` (pronóstico − media de los pronósticos de la estación a esa hora y horizonte en los 7 días previos) | curva | Los pronósticos suavizan los extremos; en valor absoluto la curva aprendería la estación del año |
  | `dep_rocio` (temperatura − punto de rocío) | curva | Cerca de la saturación la temperatura casi no baja del punto de rocío |
  | `viento_vel` | curva | Mezcla de la capa cercana al suelo (efecto con umbral) |
  | `err_ult` × horizonte (y `vec_err_1h` × horizonte, si entran los vecinos) | lineal con peso que depende del horizonte | La persistencia del error se apaga con la antelación |
  | **Enfriamiento nocturno** = noche × despejado × viento < 10 km/h | indicador | Inversión por radiación que los modelos subestiman |
  | Sesgo exponencial × noche (los 4 α) y `err_ayer` × condición | interacciones | El sesgo de la estación y la persistencia dependen de la noche y del tiempo |
  | Resto de las columnas de cada familia | lineal | — |

  - **Noche** = sol bajo el horizonte (elevación < −0,833°), calculada con fecha, hora, latitud y longitud (`elev_solar`, `noche` en `features.py`). **Calma** = viento pronosticado < 10 km/h (≈ 3 m/s, valor usual para que se forme la inversión). **Despejado** = el grupo «despejado». Los umbrales se fijaron por criterio físico y no se ajustaron con los datos.
- **Faltantes:** las columnas lineales y `temp_anom` se imputan con 0 y llevan un indicador de faltante; `prob_lluvia`, el horizonte y las curvas, con la mediana del entrenamiento. Splines, medianas y escalado se ajustan solo con el entrenamiento de cada pliegue.
- **Hiperparámetro elegido con datos:** la penalización α, en una grilla de 1 a 10⁷ (`ridge_alphas`), en los pliegues de elección. Con ~10⁶ filas estandarizadas, α chico casi no penaliza: por eso se busca en una grilla amplia.
- **Versión descartada (en `main`): Elastic Net con todas las variables.** La proporción L1/L2 no cambiaba el error, quedaban 3 de los 4 sesgos exponenciales y humedad y rocío tenían coeficientes grandes de signo opuesto. Ridge con todas las variables daba el mismo error que el Elastic Net (1,290 contra 1,286) y es más legible.

### Modelo que prioriza minimizar el error: LightGBM

- **LightGBM con las 8 familias** (51 variables). `condicion` va como categórica; incluye dirección del viento (sen/cos), latitud y longitud. No usa `station_id`: el sesgo de cada estación entra por los sesgos exponenciales. No se imputan faltantes.
- **Grilla (pliegues de elección):** `num_leaves` ∈ {31, 127} × objetivo ∈ {L2, L1}, con la cantidad de árboles en cortes cada 50, de 50 a 500. Los cortes finos se definieron en `main`, después de que una grilla de 250 a 2000 árboles dejara el óptimo en el borde: con ~20 días de entrenamiento, y las filas repetidas por horizonte, el modelo se sobreajusta rápido. Quedan fijos `learning_rate` 0,05, `min_child_samples` 500, `feature_fraction` y `bagging_fraction` 0,8, y `lambda_l2` 1 (`config.yaml`).

### Elección de hiperparámetros sin optimismo (los dos modelos)

- **Regla:** los hiperparámetros se eligen en los **pliegues de elección (1–2)** y la métrica del informe se reporta en los **pliegues siguientes (3 en adelante)**, que no se usaron para elegir. Una vez elegidos quedan **congelados** (`*_hiperparametros.json`, con las fechas de los pliegues que los eligieron); se vuelven a elegir si cambian esas fechas o las familias, o con `--reelegir`. El modelo se reentrena en cada pliegue de reporte con todo su entrenamiento.
- **Criterio de elección:** MAE medio en filas reales con historia completa.
- **Entrenamiento:** filas reales e interpoladas (decisión 5 del Paso 0), con una submuestra aleatoria fija (semilla de `config.yaml`): 10⁶ filas para el Ridge (matriz ancha, 8 GB de RAM) y 2·10⁶ para LightGBM. La evaluación usa todas las filas reales de prueba.
- **Historia completa en el reporte:** los días con historia completa de los pliegues de reporte están en los pliegues 3 y 5; los de historia incompleta, en el 4 y el 5.

### ¿Las diferencias entre modelos son reales?

Código: `src/comparar_modelos.py`. Salidas en `data/resultados/comparacion_*.csv`.

- **Unidad de la prueba: el día de prueba,** no la fila. Dentro de un día los errores de estaciones y horizontes están muy correlacionados, así que contar filas daría p-valores falsamente chicos.
- **Prueba:** t pareado sobre la diferencia de MAE diario (Diebold–Mariano con pérdida absoluta), un intervalo bootstrap remuestreando días y la cantidad de días que gana cada modelo. Con pocos días el bootstrap es optimista: manda el t.
- Resultados: en «Resultado de la experimentación y modelos finales».

### Experimentación por familias y regla de recorte del Ridge (17/11/2026)

- **Orden:** la experimentación va **antes** del modelo final y decide qué familias usa el Ridge. LightGBM se entrena siempre con las 8 familias: la pregunta del producto es si un modelo de caja negra con toda la información es sustancialmente mejor que uno regularizado y de caja de cristal. Los dos modelos finales se comparan contra el NWS crudo y los baselines.
- **Datos de los experimentos:** fecha de corte congelada (09/11/2026). Los días posteriores no reabren los experimentos.
- **Producto:** pliegues anclados al calendario. Un día nuevo agrega un pliegue y no cambia los anteriores. Los hiperparámetros se eligen una vez en los primeros pliegues y quedan congelados.
- **Recorrido por familia (1 a 8):** hipótesis experta → exploración (correlaciones, boxplots, redundancia) → Ridge con todas las familias → Ridge sin cada familia. Después se decide si conviene el formato acumulativo.
- **Regla de recorte (fijada antes de ver resultados):** una familia se saca del Ridge si, sin ella, el MAE promedio de los pliegues de elección no empeora en más de **0,005 °C** y en ningún pliegue de elección empeora en más de **0,010 °C**.
  - **Por qué antes:** si el umbral se elige mirando las tablas, se termina eligiendo el que favorece el resultado esperado. Escrita antes, la regla se puede defender, repetir y aplicar sola con datos nuevos.
  - **Por qué 0,005 °C:** es menos del 0,5 % del MAE (≈ 1,3 °C) y está por debajo del ruido entre días ya medido (en la rama `main`, sumar las familias 5 a 8 mejoró LightGBM 0,012 °C por día con p = 0,12). Lo que cambia menos que eso no se puede distinguir con estos datos y solo resta legibilidad.
  - **Por qué 0,010 °C por pliegue:** no sacar una familia que en promedio no importa, pero en un período concreto sí ayuda. Duplica el umbral promedio para tolerar el ruido de un solo pliegue.
  - **Confirmación conjunta:** las familias que la regla saca una por una se sacan juntas. Si juntas empeoran más de 0,005 °C (se reemplazan entre sí), se devuelve al modelo la de **mayor** costo de sacarla (la que más ayuda) y se vuelve a probar.
  - El grupo puede apartarse de la regla solo dejando escrita la razón.
- Diseño completo: `docs/superpowers/specs/2026-11-17-experimentos-y-pliegues-anclados-design.md`.

### Resultado de la experimentación y modelos finales (17/11/2026)

- **Datos:** 55 días (10/09 → 09/11), 48 usables. Pliegues anclados desde el 30/09: 1–2 eligen, 3–5 reportan (5 = 02/11–06/11). Los días 07–09/11 esperan a completar un bloque.
- **Ablación** (`notebooks/experimentos.ipynb`, pliegues de elección, historia completa). Empeoramiento del MAE al sacar cada familia, en °C (promedio / peor pliegue):

  | Familia | Sin ella | Decisión |
  |---|---|---|
  | base | +0,017 / +0,034 | queda |
  | sesgo EWMA | +0,015 / +0,022 | queda |
  | rezagos | +0,006 / +0,007 | queda |
  | pendiente | +0,006 / +0,007 | queda |
  | revisión | +0,002 / +0,014 | queda (en el pliegue 2 supera 0,010) |
  | rezagos extra | +0,002 / +0,003 | sale |
  | vecinos | −0,001 / −0,001 | sale |
  | dinámica | −0,001 / −0,001 | sale |

  - Sacar las tres juntas tampoco empeora: Ridge final 1,247 °C contra 1,246 con todas (elección).
  - Rezagos extra mueve la predicción (aporte 0,16 °C), pero repite a los rezagos (redundancia 0,87): por eso sacarla no cambia el error.
  - El grupo no se apartó de la regla.
- **Ridge final:** base, pendiente, rezagos, sesgo EWMA y revisión. α = 10⁵ (elegido en los pliegues 1–2; queda congelado).
- **LightGBM:** las 8 familias. 127 hojas, pérdida L1, 150 árboles (elegido en los pliegues 1–2; queda congelado). Importancia por familia: base 57 %, sesgo EWMA 18 %, pendiente 12 %; el resto, 0,2–4 % cada una.
- **MAE en los pliegues de reporte 3–5** (°C, todos los días de prueba):

  | Horizonte | B0 NWS crudo | B2 estación × hora | Ridge | LightGBM |
  |---|---|---|---|---|
  | todos | 1,541 | 1,441 | 1,361 | **1,310** |
  | 1 h | 1,380 | 1,295 | 1,063 | **1,034** |
  | 6 h | 1,421 | 1,326 | 1,220 | **1,154** |
  | 24 h | 1,536 | 1,430 | 1,363 | **1,311** |
  | 48 h | 1,687 | 1,601 | 1,542 | **1,521** |

  - El Ridge con 5 familias da lo mismo que con las 8 (1,361 contra 1,359): el recorte también se sostiene en los pliegues de reporte.
- **¿Las diferencias superan el ruido?** (`comparar_modelos.py`, el día como unidad, 15 días de prueba):
  - LightGBM − Ridge: −0,051 °C por día (IC95 −0,071 a −0,032), gana 14 de 15 días, p = 0,0002. Con historia completa: −0,054, 7 de 7 días, p = 0,0006.
  - Ridge − B2: −0,080 °C, 14 de 15 días, p = 0,0001. B2 − B0: −0,101 °C, 13 de 15 días.
- **Respuesta a la pregunta del producto:** la caja negra con toda la información es mejor de forma consistente, pero por poco: 0,05 °C (≈ 4 % del MAE del Ridge). El Ridge de caja de cristal ya captura la mayor parte de la mejora sobre el NWS (11,7 % contra 15,0 % de LightGBM).
- Salida completa: `salidas/modelos_finales.txt`.

### Experimento: modelo explicativo OLS con selección por RFE (17/11/2026, rama `tp3-ols-rfe`)

Segundo modelo de caja de cristal, independiente del Ridge. Código: `src/modelos/ols_rfe.py`. Lectura: `notebooks/rfe_ols.ipynb`. Salidas: `salidas/ols_rfe/`.

- **Qué cambia:** solo el método. El dataset y la matriz de diseño son los mismos (`Disenio` con las 8 familias). LightGBM, `features.py` y el Ridge del producto no se tocan.
- **Independencia del Ridge:** no se usan su α, su ablación ni sus 5 familias. Los resultados del Ridge y de LightGBM se leen recién en la tabla final.
- **Estimador:** regresión lineal sin penalización (OLS). La matriz es singular (las bases spline y los indicadores suman 1), así que se resuelve con la seudoinversa. La predicción y el aporte de cada variable son únicos; los β sueltos, no.
- **Unidad que se elimina:** la variable de `features.py`, con todas sus columnas de diseño. Son 50 variables.
- **Orden:** RFE clásico. En cada paso se reentrena y sale la variable con menor desvío de su aporte Σβx en el entrenamiento. El peso se promedia entre los pliegues 1–2, así hay un solo orden.
- **Cuántas quedan:** el conjunto más chico que, contra el mejor punto de la curva, no empeora más de 0,005 °C de promedio ni 0,010 °C en ningún pliegue de elección. Son los umbrales del grupo.
- **Resultado en elección:**
  - OLS con las 50 variables: 1,281 °C.
  - Mejor punto: 31 variables, 1,270 °C.
  - **Elegido: 21 variables**, 1,274 °C.
  - La curva es irregular por debajo de ~15 variables.
- **Las 21:** 10 de base, los 4 sesgos EWMA, `pendiente_pron`, `err_ayer`, `rev_12h`, `err1h_ult` y 3 de dinámica (`obs_tend3h`, `err1h_tend3h`, `salto_obs`). **Sale `err_ult`**: su peso lo absorbe `err1h_ult`, que dice casi lo mismo.
- **MAE en los pliegues de reporte 3–5** (°C, todos los días):

  | Horizonte | B2 | OLS + RFE | Ridge | LightGBM |
  |---|---|---|---|---|
  | todos | 1,441 | 1,369 | 1,361 | **1,310** |
  | 1 h | 1,294 | 1,150 | 1,063 | **1,034** |
  | 6 h | 1,325 | 1,224 | 1,220 | **1,154** |
  | 24 h | 1,430 | 1,368 | 1,365 | **1,312** |
  | 48 h | 1,598 | 1,542 | 1,539 | **1,517** |

- **Por día (15 días):**
  - Contra el Ridge, OLS + RFE pierde por +0,008 °C (IC95 +0,000 a +0,015). Gana 5 de 15 días, p = 0,064. Con historia completa empatan (+0,002, p = 0,81).
  - Contra B2 gana 13 de 15 días.
- **Lectura:** con 21 variables queda prácticamente empatado con el Ridge en todos los horizontes salvo 1 h, donde pierde 0,09 °C. Esa pérdida es el `err_ult` que RFE sacó. Los 4 sesgos EWMA cambian de signo entre pliegues (33 % de sus columnas): son colineales y OLS no los encoge.
- **Decisión:** pendiente del grupo. El Ridge del producto no cambia.

### Estructura: modelos experimentales y modelo producto (17/11/2026)

- **Experimental:** cada tipo de modelo es un archivo en `src/modelos/` (`ridge.py`, `lgbm.py`, `ols_rfe.py`). Ahí se eligen sus hiperparámetros en los pliegues de elección y se reporta en los de reporte. No los corre el pipeline.
- **Producto:** `config.yaml` (`producto.modelo`) dice cuál es. Lo decide el grupo con `notebooks/comparacion_modelos.ipynb`. Hoy es **LightGBM**, el de menor error.
- `src/producto.py`, paso 7 del pipeline, entrena el producto con todos los días usables y lo guarda en `data/producto/` con su ficha (hiperparámetros, días, MAE de reporte). No elige nada: si faltan los hiperparámetros congelados o son viejos, se detiene y dice qué script experimental correr.
- Cambiar de producto es cambiar una línea de `config.yaml`, sin tocar código. Todos los modelos tienen la misma interfaz (`src/modelos/__init__.py`).
- **Salidas por modelo:** cada modelo escribe todo lo suyo en `salidas/<modelo>/` (`ridge/`, `lgbm/`, `ols_rfe/`), versionado: hiperparámetros congelados, elección, métricas por pliegue, coeficientes o importancias y modelos. Los experimentos viejos (`--todas`, `--vecinos`) van en `salidas/<modelo>/historico/`. Lo común (baselines, comparación, interpretación) queda en `data/resultados/`. Los nombres de archivo de la tabla de abajo (`interpretable_*`, `lgbm_*`) ahora están sin el prefijo, dentro de la carpeta de su modelo.

### Decisiones de implementación y su costo (17/11/2026)

Decisiones tomadas al implementar la experimentación y los pliegues anclados (rama `tp3-experimentos`), revisadas por el grupo. «Costo» = qué se pierde si la decisión resultara equivocada.

| # | Decisión | Por qué | Costo si está mal |
|---|---|---|---|
| 1 | El Ridge elige α con una sola matriz XᵀX por pliegue (`src/ridge_gram.py`), no reentrenando scikit-learn por cada α | Mismo resultado (test contra scikit-learn) y las 8 ablaciones tardan minutos en lugar de ~3 h | Elegir un α distinto al de scikit-learn; lo cubre el test |
| 2 | Función `contribuciones()`: aporte de cada familia fila por fila | La necesita el control de que los aportes suman la predicción | Ninguno |
| 3 | Grilla de α del Ridge de 1 a 10⁷ y submuestra de 1 M de filas; los 4 sesgos EWMA entran como columnas (ya no se elige uno) | Es la especificación del Ridge con todas las familias; la matriz es ancha y hay 8 GB de RAM | Grilla distinta a la anterior |
| 4 | En interpretación y dashboard el Ridge entrena con su propia submuestra de 1 M de filas (LightGBM, con 2 M) | Con 8 GB de RAM, la matriz ancha con 2 M de filas no entra | El Ridge usa menos filas que LightGBM (igual que en el reporte) |
| 5 | Redundancia entre familias = mayor correlación de Spearman entre variables de cada par de familias | La exploración va antes que los modelos | No capta redundancia conjunta (solo de a pares); es descriptiva y no decide |
| 6 | Todas las variantes × α se calculan en una sola pasada por pliegue | Memoria | Ninguno |
| 7 | Funciones `mapa_correlaciones()` (variables × variables) y `evaluar_detalle()` (MAE por horizonte y segmento) | El mapa de correlaciones lo pidió el grupo; la tabla final pide el detalle | Ninguno |
| 8 | Las figuras de `experimentos.ipynb` se muestran desde el PNG guardado | `interpretar.py` fija el backend de matplotlib en `Agg` y los gráficos no aparecían | Ninguno |
| 9 | No se versiona el log de nbconvert | La salida queda en las celdas de la notebook | Ninguno |
| 10 | `interpretable_coeficientes.csv` es solo numérico; la familia de cada columna va en `interpretable_familia_de_columna.csv` | `comparacion_modelos.ipynb` lo lee como numérico | Ninguno |
| 11 | `interpretable_eleccion.csv` y el JSON de hiperparámetros conservan su formato anterior (`sesgo` = «los 4 (lineales)»); la grilla completa va en `interpretable_grilla_alpha.csv` | No reescribir celdas de `comparacion_modelos.ipynb` | La columna `sesgo` de ese CSV no es el error medio (igual que antes) |
| 12 | En `comparacion_modelos.ipynb` solo se agregó una nota arriba | No reescribir celdas existentes | El gráfico «Mejora sobre B0 por pliegue» muestra los pliegues 1–4, no el 5 (sí está en las tablas) |
| 13 | El arreglo de la clave de caché de LightGBM no tiene test propio | Es la misma lógica que la del Ridge, que sí tiene test | LightGBM reentrena de más después de un hueco en los datos |
| 14 | A los JSON de hiperparámetros ya elegidos se les agregaron las fechas de los pliegues de elección, en lugar de volver a elegir (24 min) | Se eligieron con exactamente esos pliegues (fechas verificadas) | Ninguno: reentrenado con resultados idénticos |

**Arreglos de la revisión independiente** (cada uno con un test que falló antes del arreglo):
- La clave de caché de cada pliegue solo incluye los días con historia incompleta de **su** bloque de prueba: un día nuevo después de un hueco ya no invalida los pliegues anteriores.
- La clave incluye la huella del código del modelo y las columnas de cada familia: editar el modelo ya no sirve resultados viejos.
- Los hiperparámetros congelados guardan las fechas de los pliegues de elección: si aparece tarde un día y los bloques se corren, se vuelven a elegir.
- El spec decía devolver la familia de menor costo al confirmar el recorte; el código devuelve la de mayor. Se corrigió el spec.

**Pendientes conocidos (no se arreglaron):**

| Pendiente | Efecto |
|---|---|
| `interpretar.py` asume que el Ridge tiene base, pendiente y rezagos | Falla si una regla futura saca alguna de esas familias |
| La ablación «sin base» conserva noche, condición y horizonte a través de las interacciones | «Sin base» no es del todo puro; no cambia la decisión (sacarla empeora 0,017 °C) |
| α se elige en float64 y el reporte ajusta scikit-learn en float32 | Despreciable con α = 10⁵; no garantizado con α = 1 |
| Pico de memoria de ~3–4 GB al transformar 1 M de filas de entrenamiento | Es el paso con más riesgo en 8 GB de RAM |
| `comparar_modelos.py` no verifica que el LightGBM guardado corresponda a los datos actuales | Si solo se reentrena el Ridge, compara un Ridge nuevo con un LightGBM viejo; el orden del pipeline lo evita |

## Etapa B — Interpretación

Código: `src/interpretar.py`. Figuras en `figuras/interp_*.png` y tablas en `data/resultados/interp_*.csv`. Modelos finales (Ridge con 5 familias, LightGBM con 8), pliegues de reporte 3–5. Las curvas y el SHAP salen del último pliegue, que tiene el entrenamiento más largo. SHAP: contribuciones exactas de LightGBM (TreeSHAP) en 100 000 filas de prueba.

- **Curvas del Ridge** (`interp_curvas_ridge.png`): aporte de cada variable a la corrección, en °C.
  - **Enfriamiento nocturno:** −0,81 °C en noches despejadas y calmas.
  - **Pendiente del pronóstico:** casi lineal, de −0,8 a +0,9 °C. Si el pronóstico sube rápido, la observación queda por encima.
  - **Hora solar:** −0,4 °C a las 15 h y +0,2 °C de madrugada, además del sesgo exponencial.
  - **Depresión del rocío:** +0,3 °C cerca de la saturación.
  - **Viento:** de −0,3 °C en calma a +0,2 °C con viento fuerte.
  - **Anomalía:** chica (−0,1 a +0,3 °C); sube cuando el pronóstico es mucho más frío que el de la semana previa.
  - **Peso de `err_ult`:** 0,31 °C por °C a 1 h, 0,10 a 6 h, 0,06 a 24 h y 0,03 a 48 h.
- **SHAP de LightGBM** (`interp_shap_lgbm.png`), |contribución| media en °C: pendiente del pronóstico 0,23, viento 0,15, sesgo exponencial (α 0,3) 0,15, condición 0,12 y probabilidad de lluvia 0,12.
- **Latitud y longitud** (`interp_latlon.png`): su aporte medio por estación casi no se relaciona con el sesgo histórico de la estación (r = 0,23). Los sesgos exponenciales, en cambio, sí (r = 0,78). Lat y lon no funcionan como identificador de la estación: aportan un efecto regional.
- **¿Se captó el sesgo sistemático?** (`interp_sesgo_hora.png`):

  | Corrección | Amplitud del ciclo por hora (°C) | Desvío entre estaciones (°C) | Error medio (°C) |
  |---|---|---|---|
  | NWS crudo (B0) | 2,62 | 0,49 | +0,05 |
  | B2 | 0,49 | 0,35 | +0,09 |
  | Ridge | 0,41 | 0,32 | +0,03 |
  | LightGBM | 0,33 | 0,33 | +0,05 |

  - Los modelos casi eliminan el ciclo diario del error del NWS.
  - El desvío restante entre estaciones incluye el ruido de promediar solo 15 días de prueba, así que no todo es sesgo sin corregir.
- **Cuánto del error se anticipa** (`interp_techo.png`), como correlación entre la corrección y el error del NWS:
  - **Modelos:** 0,63–0,65 a 1 h, 0,50–0,56 a 6 h y 0,37–0,40 a 48 h. Explican entre el 14 % y el 42 % de la varianza del error.
  - **`err_ult` solo:** 0,52 a 1 h, cae a ~0 en 6–12 h y oscila con el ciclo diario (0,25 a 24 h, 0,17 a 48 h).
- **Pendiente para el informe:** elegir qué figuras van y redactarlas.

## Etapa C — Dashboard (10/11/2026)

Código: `app/dashboard.py`, con los datos de `src/preparar_dashboard.py`. Se corre desde `TP3/` con `.venv/bin/streamlit run app/dashboard.py`.

- **Herramienta:** Streamlit con Plotly, en Python como el resto del pipeline. Se entrega como código ejecutable.
- **Que cualquiera pueda correrlo** (pedido del 10/11/2026): los datos del dashboard se versionan en `app/datos/` (~26 MB, comprimidos con zstd). Así el dashboard corre sin los datos crudos y sin el pipeline. La app solo necesita `app/requirements.txt` (5 paquetes con versión fija, Python ≥ 3.11), y los pasos están en el README.
  - **Para actualizarlo con datos nuevos:** correr el pipeline completo, los modelos y después `src/preparar_dashboard.py`.
- **Identidad visual: propuesta A, de taste-skill**, estilo «libro contable acromático». Se eligió frente a la B (Carbon de IBM, del catálogo awesome-design-md).
  - **Paleta:** fondo gris azulado #f2f4f7, tarjetas blancas, tinta #0a0a0a y gris #6b6b6b.
  - **Color:** el único color es el azul #1d5fbf del pronóstico corregido y su banda. Verde y rojo marcan solo la mejora o el empeoramiento.
  - **Tipografía:** Geist.
  - **Archivos:** los tokens están en `.streamlit/config.toml` y en el CSS de la app.
- **Pregunta de la pantalla:** ¿cuánto confiar en el pronóstico y cuánto lo mejora la corrección?
- **Contenido:**
  - **Indicadores:** error del NWS, error corregido, mejora y cobertura de la banda.
  - **Serie de una estación:** observado, pronóstico del NWS y corregido con su banda, para un horizonte elegido.
  - **Gráficos de error:** por horizonte y por hora del día.
  - **Mapa:** mejora por estación.
  - **Lista:** las estaciones donde la corrección menos ayuda.
  - **Descarga:** las filas filtradas, hasta 50 000.
- **Filtros:** corrección (Ridge o LightGBM), horizonte, hora solar local, condición del tiempo y estados.
- **Datos:** solo los días de prueba de los pliegues de reporte (3–5), que los modelos no vieron al entrenar. Son 6,8 millones de filas con observación real de 516 estaciones.
- **Confianza del pronóstico** (banda empírica):
  - **Calibración:** percentiles 10 y 90 del error que queda después de corregir, medidos en las pruebas de los pliegues de elección (1–2), por tramo de horizonte × condición del tiempo. Si un grupo tiene menos de 2000 filas, se usa solo el tramo.
  - **Cobertura real en los pliegues de reporte:** 74–75 % (Ridge 0,737, LightGBM 0,750), contra el 80 % esperado. El dashboard muestra la cobertura real junto a la esperada.
  - **Sin niveles de confianza:** se descartó la etiqueta «alta / media / baja» (en `main`, el ancho de la banda variaba poco: 3,1–4,7 °C según horizonte y condición).
- **Hora solar en lugar de hora civil:** no tenemos la zona horaria de cada estación, y AL, MS y parte de TN, KY y FL están en la hora Central. El filtro y los gráficos usan la hora solar local, la misma que usan los modelos, y se rotula así.
- **Límite conocido:** al cambiar los filtros, la estación elegida puede volver a la primera de la lista.

### Pestañas de comparación de modelos (17/11/2026)

- **Pedido:** comparar el Ridge (Gram) y LightGBM con sus versiones, variables, búsqueda de hiperparámetros y regularización. El dashboard pasa a tener 8 pestañas: **Hallazgos** (la narrativa que pide la consigna), **Resumen**, **Ridge (Gram)**, **LightGBM**, **Ridge vs LightGBM**, **Monitoreo** (la vista anterior, sin cambios de contenido), **Variables** y **Datos y validación** (días excluidos, pliegues walk-forward, baselines B0–B3 y MAE por pliegue).
- **Datos de las pestañas nuevas:** `src/preparar_resultados_dashboard.py` lee las salidas versionadas (`salidas/modelos_finales.txt`, `log_lgbm.txt`, `log_ridge_todas_familias.txt`) y escribe tablas chicas en `app/datos/resultados/`. También se leen `salidas/experimentos/*.csv` y las figuras de `figuras/` y `salidas/experimentos/figuras/`. Así siguen sin hacer falta `data/` ni el pipeline.
- **Color por modelo, fijo en todas las pestañas:** Ridge azul #1d5fbf, LightGBM ámbar #c56a00 (validados para daltonismo: ΔE 26,8); NWS y baselines en gris. En Monitoreo, la línea y la banda corregidas toman el color del modelo elegido.
- **Versiones anteriores (rama `main`):** se muestran en tablas con la aclaración de que se midieron con otros pliegues y no son comparables con las finales.
- **Límite:** de la grilla de LightGBM solo están versionadas las 5 mejores combinaciones (log), no la curva completa por cantidad de árboles (`data/resultados/lgbm_eleccion.csv` no está en el repo).
