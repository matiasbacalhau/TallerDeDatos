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
| base | `temp_pron`, `hum_pron`, `rocio_pron`, `dep_rocio`, `viento_vel`, `viento_sin`, `viento_cos`, `viento_calmo`, `prob_lluvia`, `condicion` (7 grupos), `horizonte`, `hora_solar_sin/cos`, `lat`, `lon` |
| pendiente | `pendiente_pron` = (pronóstico de T+1 con h+1 − pronóstico de T−1 con h−1) / 2, del **mismo ciclo de emisión** (verificado: 100 % de las filas coinciden en la hora de generación, diferencia ≤ 5 min) |
| rezagos | `err_ult`, `err_media3`, `err_media6` (mismo horizonte, hasta T−h−2) y `err_ayer` (mismo horario, T−24·k, con 24·k ≥ h+2) |
| sesgo_ewma | `sesgo_ewma_05/10/20/30`: promedio exponencial (α = 0,05/0,1/0,2/0,3) de la media diaria del error de la estación a esa hora, hasta el día D−k. α se calibra en validación |

- **Causalidad:** la fuente más reciente de cualquier variable es la hora objetivo − horizonte − 2. Control sobre las 20,7 M de filas: 0 violaciones.
- **Recálculo independiente** con pandas de 1.500 filas (25 estaciones): 0 discrepancias en las 9 variables derivadas.
- **Faltantes:** `err_ult` 31,4 %, `err_media3` 28,9 %, `err_media6` 25,7 %, `err_ayer` 32,5 %, `sesgo_ewma_*` 14,7 % (los primeros días no tienen historia); el resto, 0 %. LightGBM los tolera; el modelo lineal necesitará imputar.
- **Viento:** dirección faltante (18,8 % de las filas) equivale a velocidad 0 km/h: se trata como calma (`viento_calmo`, sen/cos = 0).
- **Condición:** 99,5 % en 6 grupos; 'otro' (0,5 %) son las heladas (`Frost`). Tormenta incluye `T-storms`, huracán y tormenta tropical.
- **Asociación exploratoria (todas las filas reales, no un resultado de modelo):** correlación de `pendiente_pron` con el error = 0,35 (h = 6: 0,38; h = 24: 0,35; h = 48: 0,27); de `err_ult` = 0,04 a 6 h, 0,28 a 24 h y 0,20 a 48 h. Es coherente con el desfase de ~37 min de la observación, pero la correlación no prueba la causa.
