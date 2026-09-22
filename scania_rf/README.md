# Modelo Random Forest — Falla del sistema APS (Scania)

Este proyecto predice fallas del sistema de aire a presión (APS) de camiones Scania a partir del dataset público [APS Failure at Scania Trucks](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks), optimizando una función de costo de negocio que penaliza más los falsos negativos (fallas no detectadas) que los falsos positivos (revisiones innecesarias).

El notebook original (`Scania_random_forest_buenas_practicas.ipynb`) fue separado en 3 scripts, siguiendo el mismo esquema usado en el hands-on de la Clase 1 (ETL → entrenamiento → testeo), como parte del Track A del flujo de trabajo del TP.

## Estructura

```
.
├── aps_failure_training_set.csv   # dataset de entrenamiento (crudo)
├── aps_failure_test_set.csv       # dataset de test (crudo)
├── etl_scania.py                  # 1. limpieza y preparación de datos
├── train_scania_rf.py             # 2. entrenamiento del modelo
└── test_scania_rf.py              # 3. evaluación del modelo
```

Los tres scripts se ejecutan en orden y se van pasando información entre sí a través de archivos (CSV, un binario `.pkl` y un `.json`), no por import directo entre ellos. Cada uno se corre así:

```bash
python etl_scania.py
python train_scania_rf.py
python test_scania_rf.py
```

## 1. `etl_scania.py` — Extracción, limpieza y preparación de datos

Se encarga de dejar los datos listos para entrenar. Concretamente:

- Carga los dos CSV crudos (`aps_failure_training_set.csv` y `aps_failure_test_set.csv`).
- Elimina la columna constante `cd_000`, que no aporta información al modelo.
- Imputa los valores faltantes por la mediana de cada columna.
- Elimina, de cada par de features con correlación absoluta mayor a 0.9, una de las dos columnas (features redundantes entre sí).
- Separa el dataset de entrenamiento en train/validación (80/20, estratificado por la clase), y separa features de target en los tres conjuntos.
- Elimina las features cuya correlación absoluta con el target es menor a 0.075 (aportan poca información y encarecen el entrenamiento).

**Artefactos que genera** (quedan en la misma carpeta, y son el input de `train_scania_rf.py` y `test_scania_rf.py`):

| Archivo | Contenido |
|---|---|
| `X_train.csv`, `y_train.csv` | Features y target de entrenamiento |
| `X_val.csv`, `y_val.csv` | Features y target de validación (para elegir el umbral de decisión) |
| `X_test.csv`, `y_test.csv` | Features y target de test (evaluación final) |
| `log_etl.txt` | Columnas eliminadas en cada filtro y shapes finales de los datasets |

## 2. `train_scania_rf.py` — Entrenamiento del modelo

Lee los artefactos de `etl_scania.py` y entrena el modelo:

- Entrena un `RandomForestClassifier` con búsqueda de hiperparámetros por `GridSearchCV` (`n_estimators`, `max_depth`, `min_samples_leaf`, con `class_weight="balanced"` por el desbalance de clases).
- Usa como scorer la función de costo de negocio del problema (`calcular_costo`), en vez de una métrica genérica como accuracy o F1, para que el modelo elegido sea el que minimiza el costo real.
- Con el modelo ya entrenado, barre distintos umbrales de probabilidad sobre el set de **validación** y elige el que da el menor costo de negocio (no necesariamente 0.5).

**Artefactos que genera:**

| Archivo | Contenido |
|---|---|
| `modelo_rf.pkl` | Binario del modelo entrenado (mejor estimador del GridSearchCV) |
| `modelo_rf_info.json` | Hiperparámetros elegidos, umbral de decisión y métricas del entrenamiento (tiempo, memoria) |
| `log_entrenamiento.txt` | Detalle legible de lo anterior + tabla completa del barrido de umbrales |

> ⚠️ La búsqueda de hiperparámetros (`GridSearchCV`) prueba varias combinaciones con validación cruzada, así que puede tardar varios minutos según la máquina.

## 3. `test_scania_rf.py` — Evaluación del modelo

Toma el modelo y el umbral guardados por `train_scania_rf.py` y los evalúa sobre el set de **test** (nunca visto durante el entrenamiento ni la elección de umbral):

- Calcula sensibilidad, especificidad, exactitud balanceada, precisión, recall, F1 y el costo de negocio total.
- Genera 3 gráficos de diagnóstico como imágenes PNG.

**Artefactos que genera:**

| Archivo | Contenido |
|---|---|
| `matriz_confusion.png` | Matriz de confusión del modelo sobre test |
| `curva_roc.png` | Curva ROC, marcando el punto de operación del umbral elegido |
| `importancia_features.png` | Top 10 features más importantes según el modelo |
| `log_testeo.txt` | Todas las métricas numéricas + `classification_report` de sklearn |

## Buenas prácticas aplicadas

- Código modular: cada función resuelve una única tarea, con nombre descriptivo.
- Documentación con docstrings en todas las funciones (parámetros y retorno).
- Validado con [Ruff](https://docs.astral.sh/ruff/) (reglas `E`, `F`, `I`, `N`, `UP`, `PL`, línea máxima 88 caracteres), siguiendo la configuración usada en la cátedra.
- Sin estado oculto entre scripts: toda la comunicación entre etapas es a través de archivos explícitos (CSV, `.pkl`, `.json`), para que cada paso se pueda re-ejecutar de forma independiente.

## Requisitos

```
pandas
numpy
scikit-learn
psutil
matplotlib
seaborn
```

(ya incluidos en el entorno `uv` del proyecto — `uv sync` antes de correr los scripts).
