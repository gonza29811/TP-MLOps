# Documentación del proyecto — TP Final MLOps 1

Este documento registra las decisiones de diseño e implementación que fuimos tomando a lo largo del proyecto, y por qué las tomamos. La idea es que sirva tanto de documentación técnica del trabajo como de registro para nosotros mismos, para no perder de vista los motivos detrás de cada decisión.

## Elección de la base del proyecto

Para la parte de infraestructura containerizada partimos del repositorio `amq2-service-ml` de la cátedra, que trae armado con Docker Compose los servicios que pide la consigna: Apache Airflow, MLflow, una base PostgreSQL, MinIO (como reemplazo local de S3) y un servicio de FastAPI para servir el modelo.

Dentro de ese repositorio hay dos ramas relevantes:

- `main`: es el scaffold base, sin ningún DAG ni modelo cargado. Es el punto de partida real para instalar el entorno.
- `example_implementation`: es un ejemplo completo y funcional que arma la cátedra sobre otro dataset (Heart Disease), pensado explícitamente como guía para que cada grupo entienda cómo se estructura un DAG, cómo se loguea un experimento de búsqueda de hiperparámetros en MLflow, y cómo se conecta el modelo a la API.

Decidimos instalar el entorno nosotros mismos siguiendo el README de la rama `main` paso a paso (crear las carpetas de Airflow, ajustar el `.env`, levantar Docker Compose), y usar `example_implementation` únicamente como referencia para entender el patrón antes de escribir nuestro propio DAG y nuestro propio experimento de MLflow para el caso de Scania. No copiamos su código directamente: lo usamos para entender la forma en que hay que resolver el problema, y después lo adaptamos a nuestro propio modelo y dataset.

## Separación del notebook en scripts

El notebook original (`Scania random forest.ipynb`) contenía toda la exploración de datos, el entrenamiento y la evaluación del modelo en un único archivo. Como buena práctica de código modular, lo separamos en tres scripts independientes, siguiendo el mismo esquema que se usó en el hands-on de la Clase 1 (separar en etapas de ETL, entrenamiento y testeo):

- **`etl_scania.py`**: carga los datos crudos, limpia el dataset (elimina la columna constante, imputa valores faltantes), filtra features muy correlacionadas entre sí y con baja correlación respecto al target, separa los datos en train/validación/test, y guarda los artefactos resultantes (CSVs) para que los consuman los siguientes scripts.
- **`train_scania_rf.py`**: entrena el modelo de Random Forest con búsqueda de hiperparámetros (`GridSearchCV`), usando como scorer la función de costo de negocio del problema (en vez de una métrica genérica), y elige el umbral de decisión que minimiza ese costo sobre el set de validación. Guarda el modelo entrenado y la información del entrenamiento.
- **`test_scania_rf.py`**: evalúa el modelo final sobre el set de test (nunca visto durante el entrenamiento ni la elección de umbral), calculando las métricas de clasificación y generando los gráficos de diagnóstico (matriz de confusión, curva ROC, importancia de features).

Cada script se documentó con docstrings en todas sus funciones, y se validó con Ruff siguiendo la misma configuración que usa la cátedra (línea máxima de 88 caracteres, reglas de estilo PEP 8, entre otras), tal como pidió el profesor en la Clase 1.

## Infraestructura containerizada

Copiamos al repositorio del equipo la infraestructura de la rama `main` de `amq2-service-ml`:

- `docker-compose.yaml`, junto con los Dockerfiles de cada servicio (Airflow, MLflow, FastAPI, PostgreSQL), en la carpeta `dockerfiles/`.
- La carpeta `airflow/`, incluyendo `secrets/` (para variables y conexiones de Airflow).
- El archivo `.env` con las variables de entorno necesarias para levantar el stack (puertos, credenciales de los servicios locales, nombres de buckets).
- `.gitignore` y `.gitattributes` del scaffold, para mantener fuera del repositorio archivos como `.venv`, `__pycache__`, o los logs de ejecución de Airflow.

Siguiendo el paso 3 del README de `amq2-service-ml`, dentro de `airflow/` creamos las carpetas `config/`, `dags/`, `logs/` y `plugins/`, que Airflow necesita que existan antes de levantar los contenedores.

Una aclaración importante: la carpeta `airflow/logs/` queda excluida del control de versiones por el `.gitignore` (ahí se acumulan los logs de cada ejecución de Airflow, y no tiene sentido versionarlos). Esto significa que, al clonar el repositorio, cada integrante del equipo tiene que crear esa carpeta vacía de forma manual antes de levantar Docker Compose, porque Airflow espera encontrarla ya creada.

Respecto a las credenciales del `.env`: son las credenciales de ejemplo que trae el scaffold de la cátedra (usuario/contraseña genéricos para Airflow, PostgreSQL y MinIO). No representan un riesgo real porque los servicios solo quedan expuestos en `localhost` de la máquina de cada uno, no son accesibles desde internet.

## Verificación del entorno

Una vez levantado el stack con `docker compose --profile all up`, verificamos que los cuatro servicios quedaran operativos:

- **Airflow** (`localhost:8080`): el panel de estado muestra en verde MetaDatabase, Scheduler, Triggerer y Dag Processor. Todavía no hay DAGs propios cargados.
- **MLflow** (`localhost:5001`): accesible, con el experimento `Default` creado automáticamente.
- **MinIO** (`localhost:9001`): accesible, con los buckets `data` y `mlflow` ya creados por el contenedor de inicialización.
- **FastAPI** (`localhost:8800/docs`): accesible, mostrando por ahora el endpoint base del scaffold (`GET /`).

Con esto confirmamos que la infraestructura containerizada quedó correctamente instalada y funcionando, antes de empezar a construir el DAG y el experimento propios del proyecto.