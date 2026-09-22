# TP Final — MLOps 1 (CEIA - FIUBA)

Implementación productiva del modelo de detección de fallas del sistema de aire a presión (APS) en camiones Scania, sobre el entorno containerizado de la cátedra (Apache Airflow, MLflow, MinIO, PostgreSQL y FastAPI).

## Estructura del repositorio

- **`scania_rf/`**: el modelo de Machine Learning propiamente dicho (datos, scripts de ETL, entrenamiento y testeo). Ver [`scania_rf/README.md`](scania_rf/README.md) para el detalle de cada script.
- **`airflow/`, `dockerfiles/`, `docker-compose.yaml`, `.env`**: la infraestructura containerizada sobre la que se despliega el proyecto.
- **[`DOCUMENTACION.md`](DOCUMENTACION.md)**: registro de las decisiones de diseño e implementación tomadas a lo largo del proyecto.

## Cómo levantar el entorno

1. Instalar [Docker](https://docs.docker.com/engine/install/).
2. Clonar este repositorio.
3. Crear las carpetas `airflow/config`, `airflow/dags`, `airflow/logs`, `airflow/plugins` (esta última no se versiona en git, hay que crearla manualmente).
4. En la raíz del repositorio, ejecutar:

```bash
docker compose --profile all up
```

5. Acceder a los servicios en:
   - Apache Airflow: http://localhost:8080
   - MLflow: http://localhost:5001
   - MinIO: http://localhost:9001
   - API: http://localhost:8800/
   - Documentación de la API: http://localhost:8800/docs
