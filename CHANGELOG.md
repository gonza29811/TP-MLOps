# CHANGELOG — cambios para la entrega final

Este archivo registra, en orden cronológico, los cambios que fuimos haciendo al proyecto a partir de la devolución del profesor sobre la entrega parcial (27/09), de cara a la entrega final (18/10). La idea es que alcance con leer este archivo para entender qué cambió desde la entrega parcial sin tener que comparar commits a mano; para el detalle de por qué se tomó cada decisión de diseño, seguimos usando `DOCUMENTACION.md`.

## DAG de extracción y carga de datos (`proceso_EL`)

Atiende el punto de la devolución sobre que el flujo no arrancaba solo desde un clon limpio del repositorio: antes, para poder correr el pipeline, cada integrante tenía que ejecutar a mano `cargar_datos_originales_s3.py` para subir los datasets crudos a MinIO.

Reemplazamos el DAG `proceso_etl` por `proceso_EL` (Extract + Load, sin Transform: la limpieza de datos pasó al proceso de entrenamiento, ver la próxima entrada de este changelog). Se estructura en dos tareas:

- **`obtener_datos_crudos`**: verifica si los datasets crudos ya existen en MinIO. Si no existen, los descarga automáticamente desde el repositorio UCI ("APS Failure at Scania Trucks") y los sube al bucket. Si ya existen (por ejemplo porque alguien los subió a mano, o por una corrida anterior de este mismo DAG), los reutiliza sin volver a descargarlos. `cargar_datos_originales_s3.py` sigue existiendo como alternativa manual, pero ya no es un paso obligatorio para poner en marcha el proyecto.
- **`dividir_y_subir_splits`**: separa los datos crudos en train/validación/test (80/20 estratificado) y los sube a MinIO bajo el prefijo `datos_splits/`, sin aplicar ninguna transformación de features (sin imputar faltantes, sin descartar columnas por correlación).
