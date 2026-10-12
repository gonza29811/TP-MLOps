# Documentación del proyecto — TP Final MLOps 1

Este documento registra las decisiones de diseño e implementación del proyecto y los motivos detrás de cada una. Para levantar el entorno y ejecutar el flujo, ver el [`README.md`](README.md); para el detalle de qué cambió en la entrega final, ver el [`CHANGELOG.md`](CHANGELOG.md).

## Elección de la base del proyecto

La infraestructura containerizada parte del repositorio `amq2-service-ml` de la cátedra, que trae armado con Docker Compose los servicios que pide la consigna: Apache Airflow, MLflow, una base PostgreSQL, MinIO (como reemplazo local de S3) y un servicio de FastAPI.

Dentro de ese repositorio hay dos ramas relevantes:

- `main`: es el scaffold base, sin ningún DAG ni modelo cargado. Es el punto de partida real para instalar el entorno.
- `example_implementation`: es un ejemplo completo y funcional que arma la cátedra sobre otro dataset (Heart Disease), pensado como guía para entender cómo se estructura un DAG y cómo se loguea un experimento de búsqueda de hiperparámetros en MLflow.

El entorno se instala a partir de la rama `main`, y `example_implementation` se usa únicamente como referencia para entender el patrón antes de escribir los DAGs propios para el caso de Scania. Su código no se copió: sirvió para entender cómo resolver el problema y luego se adaptó al modelo y al dataset del proyecto.

## Infraestructura containerizada

Del scaffold de `amq2-service-ml` se conservan:

- `docker-compose.yaml`, junto con los Dockerfiles de cada servicio (Airflow, MLflow, FastAPI, PostgreSQL), en la carpeta `dockerfiles/`.
- La carpeta `airflow/`, con `config/`, `dags/`, `plugins/` y `secrets/` (variables y conexiones de Airflow). Las carpetas vacías se versionan con un archivo `.gitkeep`, de modo que al clonar el repositorio ya existen.
- El archivo `.env` con las variables necesarias para levantar el stack (puertos, credenciales de los servicios locales, nombres de buckets).
- `.gitignore` y `.gitattributes`, para mantener fuera del repositorio archivos como `.venv`, `__pycache__` o los logs de ejecución de Airflow.

La carpeta `airflow/logs/` queda excluida del control de versiones, porque ahí se acumulan los logs de cada ejecución. Docker la crea al levantar los contenedores, así que no hace falta crearla a mano.

Airflow corre con `CeleryExecutor`, con Redis como cola de tareas, tal como viene en el scaffold. Los DAGs usan `@task.virtualenv`: cada tarea crea un entorno virtual aislado y efímero por ejecución, con únicamente las dependencias que necesita, en lugar de depender del entorno global del contenedor de Airflow. A cambio, la primera ejecución de cada tarea demora más, porque instala sus dependencias.

Respecto a las credenciales del `.env`: son las credenciales de ejemplo que trae el scaffold de la cátedra (usuario y contraseña genéricos para Airflow, PostgreSQL y MinIO). No representan un riesgo real porque los servicios solo quedan expuestos en el `localhost` de cada máquina, no son accesibles desde internet.

## Datos: extracción y carga (`proceso_EL`)

El DAG `proceso_EL` se limita a extraer y cargar los datos, sin transformarlos. Tiene dos tareas encadenadas:

- **`obtener_datos_crudos`**: verifica si los datasets originales (`aps_failure_training_set.csv` y `aps_failure_test_set.csv`) existen en el bucket `data` de MinIO, bajo `datos_originales/`. Si faltan, los descarga del repositorio UCI y los sube; si ya están, los reutiliza. Los contenedores de Airflow solo montan las carpetas de `airflow/`, no el resto del repositorio, por eso el punto de partida es siempre un bucket de MinIO. Descargar los datos desde el propio DAG permite que el flujo arranque solo desde un clon limpio, sin pasos manuales previos.
- **`dividir_y_subir_splits`**: mapea la clase (`neg`/`pos`) a 0/1, separa el target de las features y divide el set de entrenamiento original en entrenamiento y validación (80/20, estratificado por clase, con semilla fija). El set de test es el que viene en el dataset original. Los seis archivos resultantes (`X_*` e `y_*`) se suben a MinIO bajo `datos_splits/`.

**Por qué los datos se guardan sin transformar.** Estadísticas como la mediana de imputación o la correlación entre columnas dependen de los datos de entrenamiento. Si se calcularan antes de dividir, información de validación y de test se filtraría al ajuste del modelo. Por eso `proceso_EL` no imputa ni descarta columnas, y esa lógica vive en el `Pipeline` del modelo, que la ajusta únicamente con los datos de entrenamiento.

El DAG está programado para ejecutarse los lunes a las 8:00.

## Modelo: Pipeline con umbral de decisión

El modelo que se registra en MLflow no es un `RandomForestClassifier` suelto, sino un `Pipeline` de scikit-learn con tres pasos:

1. **Limpieza e imputación:** elimina la columna constante `cd_000` e imputa los valores faltantes por la mediana de cada columna.
2. **Selección por correlación:** de cada par de features con correlación absoluta mayor a 0,9 descarta una, y descarta además las features cuya correlación absoluta con el target es menor a 0,075.
3. **`RandomForestClassifier`.**

Las medianas y las columnas descartadas se calculan una única vez, al ajustar el `Pipeline` con los datos de entrenamiento, y quedan fijas dentro del modelo. Esto resuelve dos problemas: evita la fuga de información entre conjuntos (la selección se rehace dentro de cada pliegue de la validación cruzada) y garantiza que el conjunto de columnas no cambie entre corridas, porque forma parte del artefacto y no se recalcula con cada dataset.

El `Pipeline` se envuelve en un modelo `pyfunc` de MLflow (`ModeloConUmbral`) que además aplica el umbral de decisión sobre la probabilidad de la clase positiva y devuelve directamente 0 o 1. Así, cada versión registrada en el Model Registry es autosuficiente: recibe un registro crudo, con todas las columnas originales, y devuelve la decisión final, sin depender de un ETL externo, de un umbral guardado aparte ni de ningún otro archivo.

**Optimización por costo de negocio.** Una falla de APS no detectada (falso negativo) cuesta 500 y una revisión innecesaria (falso positivo) cuesta 10. Por eso el modelo no se optimiza por exactitud:

- La búsqueda de hiperparámetros (`GridSearchCV`, 5 pliegues, sobre `n_estimators`, `max_depth` y `min_samples_leaf`, con `class_weight="balanced"` por el desbalance de clases) usa el costo de negocio como métrica.
- El umbral de decisión no se fija en 0,5: se prueba un rango de valores sobre el set de validación y se elige el que minimiza el costo.

## Reentrenamiento challenger/champion (`entrenamiento_challenger`)

El DAG `entrenamiento_challenger` entrena un modelo nuevo (el *challenger*) y decide si reemplaza al modelo vigente (el *champion*) en el Model Registry de MLflow. No tiene programación: se dispara a mano cuando se quiere reentrenar. Tiene dos tareas:

- **`entrenar_modelo`**: descarga los splits de MinIO, ajusta el `Pipeline` con la búsqueda de hiperparámetros y la elección de umbral descritas arriba, y registra un run nuevo en el experimento `Fallas_Scania_APS` con los hiperparámetros, el umbral, las métricas de costo, falsos positivos y falsos negativos (en validación y en test), y la curva ROC y la matriz de confusión como artefactos. El modelo `pyfunc` queda guardado en el run, pero todavía no en el Model Registry.
- **`comparar_y_promover`**: decide si el challenger reemplaza al champion. Evalúa a ambos sobre el mismo set de test y compara el costo de negocio por muestra (el costo total dividido por la cantidad de muestras), para que la comparación sea justa aunque el tamaño del test cambie entre corridas. Si el challenger tiene menor costo, registra una nueva versión de `modelo_scania_fallas` y le asigna el alias `champion`. Si no, no se registra nada y el run queda solo en el historial del experimento.

Cuando todavía no existe ningún champion (primera corrida), el challenger se registra y se promueve directamente.

Para poder volver atrás sin buscar en el historial de versiones, cada promoción reasigna dos alias: `champion` pasa a la versión nueva y `champion_anterior` queda en la que estaba vigente justo antes.

## Servicio de predicción online (FastAPI)

**Online vs batch.** Se eligió servir el modelo de forma *online*: la API responde cada pedido en el momento, con el champion vigente. La alternativa, *batch*, consiste en predecir sobre un conjunto grande de registros con un proceso programado (por ejemplo, un DAG que corra cuando aparezca un champion nuevo) y guardar el resultado; es útil para volúmenes grandes, pero deja a la API sin función. Un DAG batch se podría sumar más adelante reutilizando la misma carga del modelo.

**Funcionamiento.** `dockerfiles/fastapi/app.py` carga `models:/modelo_scania_fallas@champion` desde el Model Registry. En cada pedido consulta qué versión tiene el alias `champion` y, si cambió, vuelve a cargar el modelo, de modo que una promoción hecha por `comparar_y_promover` se refleja sin reiniciar el servicio. Si todavía no existe un champion, responde 503.

**Entrada.** Como el pyfunc recibe el dato crudo, el pedido es una lista de registros con los nombres de columna originales. Las columnas ausentes se completan con valores faltantes, que el pipeline imputa con las medianas de entrenamiento. La API llama directamente al pipeline guardado dentro del pyfunc (`predict_proba`) para devolver la probabilidad además de la decisión; así evita la validación de la *signature* del modelo, que se infirió sobre las primeras 100 filas de entrenamiento y puede tipar como enteras las columnas que no tienen faltantes en esa muestra, lo que haría fallar un pedido con valores faltantes en ellas.

**Verificación.** `dockerfiles/fastapi/probar_api.py` envía una muestra de 200 registros del set de test (30 con falla). Resultado: las 30 fallas detectadas y 6 falsas alarmas sobre 170 registros sin falla, en línea con el costo medido en el entrenamiento.
