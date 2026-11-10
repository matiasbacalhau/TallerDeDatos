# TP3 — Registro de decisiones

Se actualiza a medida que el grupo decide. Sirve de base para el informe y la defensa.

## Paso 0 — Datos (2/11/2026)

| # | Decisión | Elegido | Por qué / evidencia |
|---|---|---|---|
| 1 | Dataset | El del grupo: 35 CSV diarios de `../archivos/` (10/9 → 14/10) | La consigna pide el «dataset fusionado del TP1». |
| 2 | Variable objetivo | Temperatura, `error_c = observado − pronosticado` | Es la variable del TP2. |
| 3 | Horizontes | Los 48 (1–48 h) | Decisión del grupo; 20,7 M de filas. Para la búsqueda de hiperparámetros se submuestrea. |
| 4 | Observación faltante | Interpolación lineal en el tiempo si el hueco dura ≤ 3 h; cada valor interpolado queda marcado (`obs_interpolada`) | Prueba de ocultamiento sobre lecturas reales: error absoluto medio 0,43 °C (hueco de 1 h), 0,55 (2 h), 0,71 (3 h). Recupera el 10,5 % de las estación-hora; queda 17,7 % sin dato. |
| 5 | Evaluación | Baselines: se evalúan con filas reales e interpoladas. Modelos interpretable y de mínimo error: se entrenan con ambas y se evalúan **solo en filas reales**. La tabla final compara todo sobre filas reales | Las métricas no dependen de valores estimados. |

Resultado: `data/pares/*.parquet` (formato largo: una fila por estación × hora objetivo × horizonte), generado con `src/preparar_datos.py`.

## Etapa A — Variables (en curso)

- **Restricción de causalidad (lógica, no negociable):** un error rezagado solo puede usarse si se observó a una distancia ≥ horizonte respecto de la hora objetivo, es decir, hasta la emisión (`forecast_generated_at_utc`). Se verificará con una prueba automática.
- **Ventanas de rezagos elegidas con nuestros datos.** Autocorrelación del error del pronóstico a 24 h (por estación, sin el sesgo medio, solo observaciones reales; 35 días, 840 horas por estación):

  | Distancia | 1 h | 2 h | 3 h | 6 h | 9 h | 12 h | 18 h | 24 h | 36 h | 48 h |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Autocorr. | 0,84 | 0,69 | 0,54 | 0,19 | −0,03 | −0,10 | 0,06 | 0,35 | −0,15 | 0,24 |

  Persistencia real de unas 6 h; los picos a 24 y 48 h son una interpretación (ciclo diario del error), no comprobada. Cada ventana debe justificarse además por una ablación en validación.
- **Sesgo persistente de la estación:** promedio exponencial de días (como el decaying average del NBM del NWS); α a calibrar en validación porque solo hay 35 días.

## Etapa A — Validación (2/11/2026)

- **Walk-forward (expanding window) con gap**, sin bloque final reservado (decisión del grupo; se explicó que así todo el período se usa para elegir y no queda una prueba limpia).
- Cortes por **día de la hora objetivo**, nunca por filas sueltas: cada hora objetivo aparece hasta 48 veces (una por horizonte) con la misma observación.
- Parámetros en `config.yaml` (`validacion:`): hoy 4 pliegues, bloques de 5 días, gap de 2 días y mínimo de 10 días de entrenamiento. Con más de 60 días: subir a 5–6 pliegues y bloques de 7 días.
- Código: `src/validacion.py` (`pliegues_walk_forward`, `verificar`). Con los 35 días actuales: entrenamiento de 13, 18, 23 y 28 días, y pruebas del 25/9–29/9, 30/9–4/10, 5/10–9/10 y 10/10–14/10.
- **Pendiente:** al llegar a hiperparámetros, decidir cómo evitar optimismo al elegir y reportar sobre los mismos pliegues (p. ej. elegir con los primeros pliegues y reportar el último).
- **No** se agrega validación dejando estaciones afuera (por ahora).

## Datos que crecen

El grupo actualiza `archivos/` periódicamente (> 60 días al momento del informe).
- `src/preparar_datos.py` es **incremental**: procesa solo los CSV nuevos y reescribe un día de `pares/` únicamente si cambió su observación (p. ej. el día anterior, cuando un hueco se cierra con datos del día nuevo). `--rehacer` fuerza todo de cero.
- Controlado el 2/11: segunda corrida sin cambios (0 días reescritos) y reproceso de un día simulado como nuevo dan el mismo total (20.688.083 filas, suma de `error_c` −600.328,13).
- Antes del informe: **congelar una fecha de corte** y repetir con ese corte las mediciones hechas con 35 días (autocorrelación, α, ablación).

## Etapa A — Baselines (2/11/2026)

- **Hora local = hora solar** (hora UTC + longitud/15, redondeada). No tiene el salto del cambio de hora civil del 1/11/2026 y sigue el ciclo del sol. En el dashboard se convierte a hora civil para mostrarla.
- **Baselines (la consigna dice «modelos baseline», en plural):** B0 NWS crudo (c = 0), B1 sesgo global, B2 sesgo por estación × hora solar, B3 persistencia del último error conocido. Un tramo por horizonte en B2 no mejoró (1,398 contra 1,396) y se descartó.
- **Aprenden solo con observaciones reales del entrenamiento.** Se evalúan con filas reales y con reales + interpoladas, siempre sobre las **mismas filas** (donde B2 y B3 están definidos; 75–79 % de las filas de prueba).
- **Causalidad de B3:** la observación de la hora H puede llegar hasta H + 59 min y el pronóstico se genera entre 60 min antes y 20 min después de «objetivo − horizonte». Por eso el último error usable es el de la hora objetivo − horizonte − 2. Una primera versión usaba objetivo − horizonte y daba 0,703 °C a 1 h: era leakage (el valor correcto es 1,24). `prueba_causalidad` lo controla en cada corrida (14,2 M de filas, 0 violaciones).
- Código: `src/baselines.py`. Resultados: `data/resultados/baselines_por_pliegue.csv` y `baselines_resumen.csv`.

MAE medio de los 4 pliegues (°C, filas reales; con reales + interpoladas los valores difieren en ≤ 0,01):

| Horizonte | B0 NWS crudo | B1 global | B2 estación×hora | B3 persistencia |
|---|---|---|---|---|
| todos | 1,474 | 1,475 | **1,389** | 1,943 |
| 1 h | 1,334 | 1,337 | 1,263 | **1,244** |
| 3 h | 1,354 | 1,356 | **1,276** | 1,585 |
| 6 h | 1,372 | 1,373 | **1,292** | 1,878 |
| 12 h | 1,390 | 1,395 | **1,331** | 2,027 |
| 24 h | 1,488 | 1,489 | **1,393** | 1,754 |
| 48 h | 1,604 | 1,600 | **1,511** | 2,008 |

RMSE (todos): 1,868 / 1,870 / 1,767 / 2,491. Sesgo residual: −0,012 / −0,132 / −0,105 / +0,042.

Lecturas:
- B2 mejora a B0 un 5,8 % en MAE: hay **sesgo condicional** por estación y hora aprendible.
- B1 empeora levemente el sesgo (−0,132 contra −0,012): el sesgo global aprendido en el entrenamiento (≈ +0,12) no se sostiene en la semana siguiente. El nivel medio del error cambia entre semanas.
- B3 solo supera a B0 a 1 h (1,244 contra 1,334) y pierde desde 3 h.

## Etapa A — Feature engineering (2/11/2026)

Familias elegidas para la primera versión: **1 base, 2 pendiente del pronóstico, 3 rezagos de horas, 4 sesgo exponencial por estación y hora**. Quedan para una ablación posterior: 5 región/vecinas y 6 revisión del pronóstico. Quedan fuera: mes, día del año, día de la semana (con 2–3 meses no se puede aprender un ciclo anual y el mes funcionaría como identificador del período) e identificador de la estación como variable (519 estaciones con ~28 días cada una: memorizaría; su sesgo habitual lo capta la familia 4).

Código: `src/features.py` (genera `data/features/<dia>.parquet` y `familias.json`) y `src/verificar_features.py`.

| Familia | Variables |
|---|---|
| base | `temp_pron`, `temp_anom` (agregada el 10/11), `elev_solar` y `noche` (agregadas el 10/11), `hum_pron`, `rocio_pron`, `dep_rocio`, `viento_vel`, `viento_sin`, `viento_cos`, `viento_calmo`, `prob_lluvia`, `condicion` (7 grupos), `horizonte`, `hora_solar_sin/cos`, `lat`, `lon` |
| pendiente | `pendiente_pron` = (pronóstico de T+1 con h+1 − pronóstico de T−1 con h−1) / 2, del **mismo ciclo de emisión** (verificado: 100 % de las filas coinciden en la hora de generación, diferencia ≤ 5 min) |
| rezagos | `err_ult`, `err_media3`, `err_media6` (mismo horizonte, hasta T−h−2) y `err_ayer` (mismo horario, T−24·k, con 24·k ≥ h+2) |
| sesgo_ewma | `sesgo_ewma_05/10/20/30`: promedio exponencial (α = 0,05/0,1/0,2/0,3) de la media diaria del error de la estación a esa hora, hasta el día D−k. α se calibra en validación |

- **Causalidad:** la fuente más reciente de cualquier variable es la hora objetivo − horizonte − 2. Control sobre las 20,7 M de filas: 0 violaciones.
- **Recálculo independiente** con pandas de 1.500 filas (25 estaciones): 0 discrepancias en las 9 variables derivadas.
- **Faltantes:** `err_ult` 31,4 %, `err_media3` 28,9 %, `err_media6` 25,7 %, `err_ayer` 32,5 %, `sesgo_ewma_*` 14,7 % (los primeros días no tienen historia); el resto, 0 %. LightGBM los tolera; el modelo lineal necesitará imputar.
- **Viento:** dirección faltante (18,8 % de las filas) equivale a velocidad 0 km/h: se trata como calma (`viento_calmo`, sen/cos = 0).
- **Condición:** 99,5 % en 6 grupos; 'otro' (0,5 %) son las heladas (`Frost`). Tormenta incluye `T-storms`, huracán y tormenta tropical.
- **Asociación exploratoria (todas las filas reales, no un resultado de modelo):** correlación de `pendiente_pron` con el error = 0,35 (h = 6: 0,38; h = 24: 0,35; h = 48: 0,27); de `err_ult` = 0,04 a 6 h, 0,28 a 24 h y 0,20 a 48 h. Es coherente con el desfase de ~37 min de la observación, pero la correlación no prueba la causa.

## Datos nuevos (16–29/10) y calidad (2/11/2026)

- `archivos/` pasó de 35 a 47 días (10/9 → 29/10). Los 12 archivos nuevos venían como `forecast_observation_v2_DDMMYYYY.csv`; se **renombraron al formato único** `forecast_observation_YYYY_MM_DD.csv` (solo cambió el nombre). `preparar_datos.py` acepta los dos formatos.
- **Días sin archivo:** 15/10, 27/10 y 28/10. El script los avisa y la validación usa días de calendario para el gap.
- **Falla de observaciones.** Fracción de filas sin observación emparejada (`unmatched`, normal 4–5 %):
  - 19/10: 18 % (2 horas con todas las estaciones sin observación); 20/10: 15 % (1 hora); 22/10: 44 % (10 horas con >90 %).
  - **23 al 26/10: 56–57 %.** Cada día, **272 de 519 estaciones no tienen ninguna observación** (contra 9–13 habituales), y son **las mismas 272 los cuatro días**. Están repartidas en 23 estados (FL 91 %, VT 83 %, GA 78 %… NY 25 %) y sus códigos van de `012MD` a `WINM6`, así que no es un bloque geográfico ni alfabético.
  - **Causa confirmada (10/11):** fue una falla del NWS, no de nuestra Lambda ni de la fusión.
- **Decisión (10/11, acordada con el docente):** esos días salen del análisis. La exclusión **no es una lista manual**: la hace una regla automática (sección siguiente), que los detecta sola.

## Calidad de los datos: exclusión automática de días y alarmas (10/11/2026)

**Principio:** el producto recibe datos nuevos todo el tiempo; agregar días no puede romper el pipeline ni requerir tocar código. Por eso la decisión de excluir un día sale de una regla que se aplica a cada día que llega, y queda registrada con el dato que la justifica.

- **Regla (relativa):** un día se excluye si su % de filas sin observación supera en más de **10 pp** la **mediana de los 14 días válidos anteriores** (mínimo 7 días de referencia). La referencia usa solo días válidos, así una falla larga no se vuelve «normal».
- **Por qué relativa y no un umbral fijo (X %):** el nivel normal cambia. En septiembre hay ~25 % de filas sin observación (cobertura total 72–82 %), en octubre ~7 %. Un umbral del 15 % sacaría casi todo septiembre; uno del 30 % dejaría pasar el 19 y el 20/10.
- **Alarmas (solo avisan, no excluyen):**
  - Por día: cobertura total (real + interpolada) < 70 % en un día incluido. Detecta degradaciones lentas que la regla relativa iría absorbiendo. Se mide sobre la total y no sobre la real porque la real de septiembre es 64–70 % y la alarma sonaría siempre.
  - Del sistema: menos del 70 % de días usables en los últimos 14 días de calendario.
- **Resultado con los 47 días actuales:** se excluyen **19, 20 y 22–26/10** (7 días; quedan 40). El 21/10 (7,7 % sin observación) queda. Ningún día incluido dispara la alarma por día; la del sistema sí suena (5 de 14 días usables entre 16 y 29/10: faltan además los archivos de 27 y 28/10).

  | Día | Sin observación | Referencia | Estaciones sin obs. |
  |---|---|---|---|
  | 19/10 | 21,7 % | 7,3 % | 13/519 |
  | 20/10 | 17,9 % | 7,3 % | 11/519 |
  | 22/10 | 48,1 % | 7,3 % | 18/519 (10 h de apagón) |
  | 23–26/10 | 55,9–58,3 % | 7,3 % | 281–282/519 |

- **Dónde queda documentado:** `data/calidad/REGISTRO_CALIDAD.md` (se regenera en cada corrida, con los avisos de cambios en la lista), `calidad_por_dia.csv` y `dias_excluidos.json`. Parámetros en `config.yaml` (`calidad:`).
- **Cómo se aplica:** `validacion.dias_disponibles()` quita los excluidos, así que baselines y modelos no los ven sin cambiar código. Las features se calculan igual para todos los días: las observaciones reales de un día excluido pueden servir de rezago a días posteriores.
- **Controles:** un día nuevo normal simulado queda incluido; uno con 60 % de faltantes queda excluido con aviso; un día con 65 % de cobertura que pasa la regla (contexto tipo septiembre) dispara la alarma. Una segunda corrida no reescribe nada.
- **Un solo comando:** `src/pipeline.py` corre preparación → calidad → validación → baselines → features → verificación, y se detiene si algún control falla.
- **Pendiente:** entender por qué septiembre tiene ~25 % de filas sin observación contra ~7 % en octubre.

## Baselines con todos los días disponibles (10/11/2026)

Datos: 47 días (10/9 → 29/10), 7 excluidos por calidad → **40 días**. 4 pliegues; las pruebas cubren 30/9–4/10, 5–9/10, 10–14/10 y 16/10–29/10. El último bloque tiene 5 días con datos válidos (16, 17, 18, 21 y 29/10), repartidos en dos semanas de calendario por los días excluidos y sin archivo.

MAE medio de los 4 pliegues (°C, filas reales); entre paréntesis, el valor con 35 días:

| Horizonte | B0 NWS crudo | B1 global | B2 estación×hora | B3 persistencia |
|---|---|---|---|---|
| todos | 1,486 (1,474) | 1,490 (1,475) | **1,389** (1,389) | 1,976 (1,943) |
| 1 h | 1,354 (1,334) | 1,358 (1,337) | **1,265** (1,263) | 1,279 (1,244) |
| 3 h | 1,375 | 1,378 | **1,281** | 1,619 |
| 6 h | 1,391 | 1,394 | **1,293** | 1,914 |
| 12 h | 1,406 | 1,412 | **1,330** | 2,074 |
| 24 h | 1,488 | 1,491 | **1,387** | 1,755 |
| 48 h | 1,645 (1,604) | 1,644 (1,600) | **1,543** (1,511) | 2,015 (2,008) |

RMSE (todos): 1,882 / 1,887 / 1,761 / 2,536. Sesgo residual: +0,007 / −0,073 / −0,041 / +0,062.

Lecturas:
- B2 sigue siendo el mejor baseline y ahora mejora a B0 un **6,5 %** en MAE (5,8 % con 35 días).
- B3 pierde a 1 h en el promedio de los 4 pliegues (1,279 contra 1,265 de B2), pero **es por el pliegue 4**: en los días con historia completa gana por poco (1,257 contra 1,262), igual que con 35 días. Ver la sección siguiente.
- B1 sigue sin mejorar a B0: el sesgo global no es estable entre semanas.
- El MAE por pliegue de B0 varía de 1,39 a 1,55: la dificultad cambia entre semanas, de modo que las comparaciones entre modelos deben hacerse siempre pliegue a pliegue.
- En el pliegue 4 solo el 50 % de las filas de prueba tiene B2 y B3 definidos (75–78 % en los demás): faltan rezagos porque sus fuentes caen en días excluidos o sin archivo.

## Días con historia incompleta y métricas por segmento (10/11/2026)

- **Regla:** un día válido (no excluido) tiene **historia incompleta** si falta o se excluyó alguno de los **3 días de calendario previos**. La marca **no excluye**: el target del día es bueno; lo que falta es parte de sus features con memoria (rezagos).
- **Por qué 3 días:** los rezagos llegan hasta 55 h atrás (h = 48, más 2 h de latencia, más la media de 6 h). Parámetro: `calidad.dias_historia`.
- **Para qué:** separar dos preguntas que el promedio mezcla. **Historia completa:** cuánto corrige el modelo con toda su información; es la **comparación entre modelos del informe**. **Historia incompleta:** cuánto corrige cuando le falta la historia reciente; mide la robustez ante caídas de la fuente, que en producción van a repetirse.
- **Dónde queda:** la lista de días marcados se genera en cada corrida en `data/calidad/REGISTRO_CALIDAD.md` y `historia_incompleta.json`. Los resultados de baselines (y de los modelos) llevan la columna `segmento` (todas / completa / incompleta).

MAE (°C, filas reales):

| Segmento | Pliegues | B0 NWS | B1 global | B2 estación×hora | B3 persistencia | B3 a 1 h | B2 a 1 h |
|---|---|---|---|---|---|---|---|
| completa | 1–3 | 1,476 | 1,482 | **1,393** | 1,953 | **1,257** | 1,262 |
| incompleta | 4 | 1,516 | 1,513 | **1,377** | 2,046 | 1,345 | **1,273** |
| todas | 1–4 | 1,486 | 1,490 | **1,389** | 1,976 | 1,279 | 1,265 |

Lecturas:
- B2 no usa rezagos, así que rinde igual en los dos segmentos (1,393 y 1,377). La persistencia (B3) es la que más se degrada: a 1 h pasa de 1,257 a 1,345.
- **Límite:** B3 solo se mide donde su rezago existe (las mismas filas para todos), así que el segmento «incompleta» de los baselines subestima el problema: las filas sin rezago no entran. En los modelos sí se evaluarán todas las filas, y ahí se verá cómo manejan los faltantes (LightGBM de forma nativa; el lineal con imputación más una variable indicadora).
- El segmento «incompleta» es un solo pliegue de 5 días: sirve para describir, no para concluir.

## Etapa A — Modelos (10/11/2026)

### Modelo que prioriza la interpretabilidad

- **Regresión lineal (Ridge) con curvas (splines) en pocas variables.** Cada variable aporta un coeficiente o una curva legible (scikit-learn: `SplineTransformer` + `Ridge`). La penalización es chica y fija (`ridge_alpha = 1`), solo por estabilidad numérica: con ~2 M de filas casi no influye.
- **Variables, elegidas por criterio meteorológico** (no por búsqueda en los datos):

  | Variable | Forma | Mecanismo |
  |---|---|---|
  | Sesgo exponencial por estación × hora (un α) | lineal | Error propio de la estación (altura, entorno, valle, costa) |
  | `err_ult` × horizonte, `err_media3/6`, `err_ayer` | lineal; `err_ult` con peso que depende del horizonte | Persistencia de la situación del tiempo |
  | Hora solar | curva periódica | Ciclo diario y transiciones de mañana y tarde |
  | `pendiente_pron` | curva | *Timing* de frentes y del ciclo diario; desfase de la observación |
  | `temp_anom` (pronóstico − media de los pronósticos de la estación a esa hora y horizonte en los 7 días previos) | curva | Los pronósticos suavizan los extremos; en valor absoluto la curva aprendería la estación del año |
  | `dep_rocio` (temperatura − punto de rocío) | curva | Cerca de la saturación la temperatura casi no baja del punto de rocío |
  | `viento_vel` | curva | Mezcla de la capa cercana al suelo (efecto con umbral) |
  | Condición y `prob_lluvia` | lineal | Nubosidad y enfriamiento por evaporación |
  | **Enfriamiento nocturno** = noche × despejado × viento < 10 km/h | indicador | Inversión por radiación que los modelos subestiman |

  - **Noche** = sol bajo el horizonte (elevación < −0,833°), calculada con fecha, hora, latitud y longitud (`elev_solar`, `noche` en `features.py`). **Calma** = viento pronosticado < 10 km/h (≈ 3 m/s, valor usual para que se forme la inversión). **Despejado** = el grupo «despejado». Los umbrales se fijaron por criterio físico y no se ajustaron con los datos.
  - **Fuera del interpretable** (solo LightGBM): humedad relativa y punto de rocío (los representa `dep_rocio`), dirección del viento (ver «Dirección del viento») y latitud y longitud.
- **Faltantes:** rezagos, sesgo y anomalía se imputan con 0 y llevan un indicador de faltante. El resto se imputa con la mediana del entrenamiento. Splines, medianas y escalado se ajustan solo con el entrenamiento de cada pliegue.
- **Único hiperparámetro elegido con datos:** el α del sesgo exponencial, en los pliegues de elección. El conjunto de variables se definió una vez y se corrió una sola vez.
- **Versión descartada: Elastic Net con todas las variables.** La proporción L1/L2 no cambiaba el error, quedaban 3 de los 4 sesgos exponenciales y humedad y rocío tenían coeficientes grandes de signo opuesto. Exploración en los pliegues de elección: Ridge con todas las variables daba el mismo error que el Elastic Net (1,290 contra 1,286), y la ganancia venía del efecto no lineal de la humedad, que una curva representa de forma directa.

### Modelo que prioriza minimizar el error

- **LightGBM** con todas las familias de features (maneja faltantes de forma nativa).

### Elección de hiperparámetros sin optimismo (los dos modelos)

- **Regla:** los hiperparámetros se eligen en la **primera mitad de los pliegues** y la métrica del informe se reporta en la **segunda mitad**, que no se usó para elegir. Con los 4 pliegues actuales: se elige en los pliegues 1–2 y se reporta en 3–4. Una vez elegidos, los hiperparámetros quedan fijos y el modelo se reentrena en cada pliegue de reporte con todo su entrenamiento.
- **Criterio de elección:** MAE medio en filas reales con historia completa.
- **Entrenamiento:** filas reales e interpoladas (decisión 5 del Paso 0), con una submuestra aleatoria fija (semilla de `config.yaml`) para que entre en memoria. La evaluación usa todas las filas reales de prueba.
- **Consecuencia a tener presente:** con los datos de hoy, el pliegue 4 tiene todos sus días con historia incompleta. La comparación limpia (historia completa) de la segunda mitad queda en un solo pliegue hasta que lleguen más días.

### Resultados del modelo interpretable (10/11/2026)

Código: `src/modelo_interpretable.py` (las piezas comunes a los dos modelos están en `src/modelado.py`). Salidas en `data/resultados/interpretable_*`.

- **Elección (pliegues 1–2, historia completa):** α del sesgo exponencial = **0,2** (MAE 1,237). Las cuatro opciones quedan a menos de 0,003 °C.
- **Reporte (pliegues 3–4), MAE en °C sobre filas reales, todos contra las mismas filas:**

  | Segmento | Pliegues | B0 NWS | B2 estación×hora | Lineal | 1 h: B2 / lineal | 48 h: B2 / lineal |
  |---|---|---|---|---|---|---|
  | completa | 3 | 1,555 | 1,471 | **1,402** | 1,272 / **1,049** | 1,705 / **1,638** |
  | incompleta | 4 | 1,532 | 1,399 | **1,301** | 1,277 / **1,064** | 1,508 / **1,429** |
  | todas | 3–4 | 1,544 | 1,435 | **1,351** | 1,275 / **1,057** | 1,606 / **1,533** |

  Con historia completa mejora al NWS crudo un 9,8 % y a B2 un 4,7 %. Con historia incompleta mejora a B2 un 7,0 %. Es mejor que el Elastic Net descartado en los dos segmentos (1,413 y 1,372).
- **Lo más influyente** (|β| estandarizado): enfriamiento nocturno, sesgo exponencial, pendiente del pronóstico, probabilidad de lluvia, cielo despejado y hora solar.
- **Enfriamiento nocturno:** en las noches despejadas y calmas (~15 % de las filas) el modelo resta **0,94 °C** (pliegue 3) y **1,09 °C** (pliegue 4) al pronóstico, además del resto de los términos. El signo y el tamaño son estables entre pliegues.
- **Sesgo exponencial:** 0,22–0,27 °C de corrección por cada °C de sesgo histórico. El resto lo reparten la curva horaria y los demás términos.
- **Pendiente para la Etapa B:** graficar las curvas (hora solar, `pendiente_pron`, `temp_anom`, `dep_rocio`, `viento_vel` y el peso de `err_ult` según el horizonte) con sus unidades.

### Resultados de LightGBM (10/11/2026)

Código: `src/modelo_lgbm.py`. Salidas en `data/resultados/lgbm_*`, con los modelos de los pliegues de reporte en `lgbm_pliegue_*.txt`.

- **Variables:** todas las familias de `features.py`. `condicion` va como categórica. No usa `station_id`: el sesgo de cada estación entra por los sesgos exponenciales. No se imputan faltantes.
- **Grilla (pliegues 1–2):** `num_leaves` ∈ {31, 127} × objetivo ∈ {L2, L1}, con la cantidad de árboles en cortes cada 50, de 50 a 500. Quedan fijos `learning_rate` 0,05, `min_child_samples` 500, `feature_fraction` y `bagging_fraction` 0,8, y `lambda_l2` 1 (`config.yaml`).
- **Elegido:** 31 hojas, objetivo L1 y **300 árboles** (MAE 1,218 con historia completa, contra 1,237 del Ridge en los mismos pliegues). La curva es plana alrededor del óptimo: de 250 a 500 árboles con 31 hojas y L1 la diferencia es menor a 0,002 °C, y 127 hojas con 100 árboles empata (1,218).
- **Grilla corregida, con el reporte ya visto.**
  - **Primera grilla:** cortes cada 250, de 250 a 2000. El óptimo quedó en el borde (250 árboles), y el error subía con más árboles en las cuatro configuraciones, más rápido con 127 hojas. Con ~20 días de entrenamiento, y las filas repetidas por horizonte, el modelo se sobreajusta rápido.
  - **Corrección:** se re-corrió con cortes más finos. La elección siguió usando solo los pliegues 1–2.
  - **Ya se habían visto los pliegues de reporte** de la primera grilla (MAE todas 1,324). El cambio fue por el borde, no por ese resultado, y el reporte casi no cambió (1,325).
- **Reporte (pliegues 3–4), MAE en °C sobre las mismas filas:**

  | Segmento | Pliegues | B0 | B2 | Lineal | LightGBM | 1 h: lineal / LGBM | 48 h: lineal / LGBM |
  |---|---|---|---|---|---|---|---|
  | completa | 3 | 1,555 | 1,471 | 1,402 | **1,382** | **1,049** / 1,071 | **1,638** / 1,651 |
  | incompleta | 4 | 1,532 | 1,399 | 1,301 | **1,267** | **1,064** / 1,102 | 1,429 / **1,398** |
  | todas | 3–4 | 1,544 | 1,435 | 1,351 | **1,325** | **1,057** / 1,087 | 1,533 / **1,525** |

  - **Contra el Ridge:** con historia completa mejora un 1,4 %. Contra el NWS crudo mejora un 11,1 %, y contra B2 un 6,0 %.
  - **A 1 h gana el Ridge**, que tiene explícito el término `err_ult` × horizonte.
  - **RMSE (todas):** 1,773 y 1,636 °C, contra 1,863 y 1,762 de B2.
  - **Sesgo del pliegue 3:** sigue en −0,30 °C, igual que B2. Ninguno de los dos corrige el cambio de régimen de esa semana.
- **Importancia** (fracción de la ganancia): pendiente del pronóstico 0,13, sesgos exponenciales 0,21 en conjunto, hora solar 0,10, latitud y longitud 0,10 en conjunto, `err_ult` 0,01.

### Dirección del viento (10/11/2026)

- **Hipótesis (del grupo):** el viento trae las propiedades del lugar de donde viene (mar o montaña). Por eso el efecto depende de la geografía de cada estación y se evalúa por **estación × sector** (8 sectores + calma), no con un coeficiente global.
- **Diagnóstico:** `src/diagnostico_viento.py`, solo en los pliegues de elección. Aprende sobre el entrenamiento un efecto por estación × sector, contraído hacia 0, sobre el residuo de B2, y lo mide en la semana de prueba. Resultados en `data/resultados/diagnostico_viento*.csv`.
- **Resultado:** fuera de muestra la mejora sobre B2 es como mucho 0,3 % (MAE 1,366 → 1,361). Sin contracción empeora 2,1 %. El efecto global por sector tampoco aporta (+0,1 %). Dentro del entrenamiento las diferencias entre sectores son grandes (mediana 0,54 °C por estación) y se concentran en estaciones de montaña y de costa, pero no se sostienen en la semana siguiente. Con ~23 días, cada estación vio pocos episodios de cada régimen de viento.
- **Decisión:**
  - **Modelo interpretable:** sin dirección del viento.
  - **LightGBM:** recibe dirección (sen/cos), latitud y longitud.
  - **Volver a evaluar** con más de 60–90 días de datos (mismo script).
  - **Trabajo futuro:** variables geográficas explícitas (orientación de la costa y relieve de cada estación).
