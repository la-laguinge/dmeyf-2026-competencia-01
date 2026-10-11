# dmeyf-2026-competencia-01
Esta es mi entrega a la competencia 1 para la materia DMEyF:

# Guia:
entrega_competencia_01/
├── README.md
├── requirements.txt
├── 01_preprocesamiento/
│   └── 01_generar_dataset_fe_v12.py
├── 02_optimizacion_bayesiana/
│   ├── 02_optuna_bimodal_v64.py
│   └── optuna_low.db
└── 03_modelo_final/
    ├── 03_entrenar_y_predecir_M4_final.py
    ├── 04_verificar_entrega.py
    ├── modelos/
    │   ├── lgb_seed_578777.txt
    │   ├── lgb_seed_377717.txt
    │   └── lgb_seed_717797.txt
    └── referencia/
        └── RECREACION_Candidato_B_I11_4M_Base_K11500.csv

# DMEyF 2026 — Primera Competencia: Predicción de Churn Bancario (202108)
**Autor:** Luis María Laguinge  
**Entrega Final:** `RECREACION_Candidato_B_I11_4M_Base_K11500.csv` (**103.7 M** en Leaderboard Público)

---

## 1. Arquitectura de la Solución Final (Modelo M4 - 4 Meses Base)

La solución final consiste en un **Ensamble homogéneo de 3 semillas (`578777`, `377717`, `717797`) de LightGBM**, entrenado sobre la ventana histórica de **4 meses (`202103`, `202104`, `202105`, `202106`)**, utilizando ingeniería de atributos focalizada sobre las 9 variables de mayor ganancia de información y los hiperparámetros óptimos obtenidos mediante Optimización Bayesiana restringida (`LowTop2`).

### Componentes Clave del Pipeline:
1. **Data Quality (Regla Cátedra):** Imputación a `NaN` de la variable `ccajas_depositos` en `202105` debido a la anomalía de carga detectada en ese mes (100% de valores en cero).
2. **Feature Engineering Focalizado (Top 9 Variables):** Cálculo en memoria (DuckDB + NumPy) de rezagos de orden 1 y 2 (`lag1`, `lag2`) y variaciones absolutas (`delta1`, `delta2`) verificando estricta contigüidad por cliente sobre las 9 variables núcleo:
   * `ctrx_quarter`, `mcaja_ahorro`, `mtarjeta_visa_consumo`, `mcuentas_saldo`, `mpayroll`, `cpayroll_trx`, `mprestamos_personales`, `mactivos_margen` y `Visa_mlimitecompra`.
3. **Target Binario Ampliado (`clase_peligro`):** Se entrena considerando `BAJA+1` y `BAJA+2` como clase positiva (`1`) para duplicar los casos de aprendizaje de fuga, mientras que la validación en Optuna y el corte se evalúan estrictamente contra la ganancia real de `BAJA+2` (`+$1.072.500` por acierto, `-$27.500` por estímulo).
4. **Optimización Bayesiana (`LowTop2`):** Búsqueda realizada con Optuna entrenando en `202103–202105` y evaluando la ganancia directa en `202106` en el corte `K = 11.500`, respetando los parámetros estructurales de la cátedra (`max_bin = 31`, `min_data_in_leaf = 0`, `learning_rate = 0.005`) y acotando `num_leaves` al rango `[12, 35]` para prevenir el sobreajuste.
   * **Hiperparámetros Ganadores (`LowTop2`):**
     * `n_estimators`: `2098`
     * `num_leaves`: `33`
     * `min_sum_hessian_in_leaf`: `0.0763`
     * `feature_fraction`: `0.5831`
     * `feature_fraction_bynode`: `0.2806`
     * `bagging_fraction`: `0.7040`
     * `bagging_freq`: `5`
5. **Ensamble de Semillas y Corte:** Se entrenan 3 modelos independientes sobre `202103–202106` con las semillas `578777`, `377717` y `717797`, se promedian sus probabilidades predichas para `202108` y se seleccionan los `K = 11.500` clientes de mayor probabilidad.

---

## 2. Evolución de Experimentos y Enfoques Descartados

A lo largo de la competencia se ejecutaron múltiples iteraciones documentadas en el Leaderboard Público:

### A. Qué NO funcionó (Hipótesis Descartadas)
* **Expansión por "Fuerza Bruta" (Lags y Deltas sobre las 152 variables crudas):** Generar más de 450 columnas diluyó la relevancia de las variables principales en el muestreo por nodo (`feature_fraction_bynode`), incrementando la varianza entre semillas y degradando la ganancia pública.
* **Validación con Fuga Temporal Deliberada (Train 202101–202106 / Val 202106):** Optimizar hiperparámetros evaluando sobre el mismo mes incluido en el entrenamiento indujo un sobreajuste severo (ganancia pública cayendo a `88.83 M` con desvío estándar de `4.69 M`).
* **Selectores Dinámicos de Familias de Variables dentro de Optuna:** Permitir que Optuna activara o desactivara bloques enteros de variables al mismo tiempo que buscaba hiperparámetros afectó la convergencia (`98.2 M – 102.3 M`), rindiendo por debajo del conjunto fijo de 9 variables núcleo.
* **Árboles Profundos (`num_leaves > 35`) con `min_data_in_leaf = 0`:** Superar las 35 hojas sin restricción de tamaño mínimo de hoja generó memorización de ruido local y desvíos estándar superiores a `2.70 M` entre semillas.
* **Exclusión de Variables de Haberes (`mpayroll`, `cpayroll_trx`):** Eliminar las variables de acreditación de sueldos por el efecto aguinaldo de junio restó poder predictivo en agosto (`202108`).
* **Agregado de Variables Complejas Aisladas (`Lag3`, Aceleración, Ratios):** Incorporar rezagos de orden 3 (`100.37 M`), aceleración (`101.17 M`) o ratios de uso de límite (`102.65 M`) de forma aislada sobre el modelo de 4 meses redujo la ganancia respecto del modelo Base limpio (`103.7 M – 103.9 M`).

### B. Progreso Iterativo hacia el Modelo Final
1. **Semillas Individuales en 4 Meses (`LowTop2`):** `102.81 M` de ganancia media con bajo desvío (`0.40 M`).
2. **Pruebas en 6 Meses y Combinaciones (`ComboFull`):** Alcanzaron picos altos pero presentaron sensibilidad en el borde del corte entre recreaciones en GPU.
3. **Ensamble de 3 Semillas sobre 4 Meses Base (`Modelo M4`):** Promediar las probabilidades de las 3 semillas sobre el modelo limpio de 4 meses (`202103–202106`) eliminó la varianza individual y consolidó una ganancia pública estable de **`103.7 M – 103.92 M`**, siendo elegido como entrega final por su parsimonia, solidez y reproducibilidad.

---

## 3. Instrucciones de Reproducción

### Orden de Ejecución
1. **`01_preprocesamiento/01_generar_dataset_fe_v12.py`:** Parte del dataset original de la competencia (`competencia_01.csv`), construye el target binario `clase_peligro` (`BAJA+1` y `BAJA+2` = 1) y genera `competencia_01_fe_v12_NA.csv`.
2. **`02_optimizacion_bayesiana/02_optuna_bimodal_v64.py`:** Script de la búsqueda con Optuna (`Optuna_Low`) de donde surgieron los hiperparámetros `LowTop2`. No es necesario volver a ejecutar esta búsqueda para generar el archivo final.
3. **`03_modelo_final/03_entrenar_y_predecir_M4_final.py`:** Construye las features sobre las 9 variables principales, entrena (o carga desde `modelos/`) las 3 semillas sobre `202103–202106` y genera el archivo único final `RECREACION_Candidato_B_I11_4M_Base_K11500.csv`.
4. **`03_modelo_final/04_verificar_entrega.py`:** Verifica formato (11.500 líneas enteras sin encabezado ni duplicados), pertenencia a `202108` y coincidencia exacta (`SHA256`) con el archivo entregado al bot.

### Nota sobre Reproducibilidad Exacta y Determinismo en GPU
El entrenamiento en Google Colab utiliza aceleración por GPU (`device = 'gpu'`, `n_jobs = -1`). Debido a que las reducciones paralelas de histogramas en punto flotante en GPU son asíncronas a nivel de hilos de hardware, re-entrenar desde cero en distintas sesiones de GPU puede variar ~0.4% de los clientes ubicados exactamente en la frontera del puesto `11.500`.  
Para garantizar **reproducibilidad 100.00% exacta bit a bit (`SHA256` idéntico)** en cualquier entorno, la carpeta `03_modelo_final/modelos/` incluye los 3 árboles entrenados en formato texto (`lgb_seed_*.txt`). Con `USAR_MODELOS_GUARDADOS = True` (activado por defecto), el script `03_entrenar_y_predecir_M4_final.py` reproduce exactamente el mismo archivo enviado al bot en menos de 10 segundos (y cambiando a `False` permite re-entrenar los 3 modelos desde cero).
