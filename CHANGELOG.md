# CHANGELOG — cambios para la entrega final

Este archivo registra, en orden cronológico, los cambios que fuimos haciendo al proyecto a partir de la devolución del profesor sobre la entrega parcial (27/09), de cara a la entrega final (18/10). La idea es que alcance con leer este archivo para entender qué cambió desde la entrega parcial sin tener que comparar commits a mano; para el detalle de por qué se tomó cada decisión de diseño, seguimos usando `DOCUMENTACION.md`.

## DAG de extracción y carga de datos (`proceso_EL`)

Atiende el punto de la devolución sobre que el flujo no arrancaba solo desde un clon limpio del repositorio: antes, para poder correr el pipeline, cada integrante tenía que ejecutar a mano `cargar_datos_originales_s3.py` para subir los datasets crudos a MinIO.

Reemplazamos el DAG `proceso_etl` por `proceso_EL` (Extract + Load, sin Transform: la limpieza de datos pasó al proceso de entrenamiento, ver la próxima entrada de este changelog). Se estructura en dos tareas:

- **`obtener_datos_crudos`**: verifica si los datasets crudos ya existen en MinIO. Si no existen, los descarga automáticamente desde el repositorio UCI ("APS Failure at Scania Trucks") y los sube al bucket. Si ya existen (por ejemplo porque alguien los subió a mano, o por una corrida anterior de este mismo DAG), los reutiliza sin volver a descargarlos. `cargar_datos_originales_s3.py` sigue existiendo como alternativa manual, pero ya no es un paso obligatorio para poner en marcha el proyecto.
- **`dividir_y_subir_splits`**: separa los datos crudos en train/validación/test (80/20 estratificado) y los sube a MinIO bajo el prefijo `datos_splits/`, sin aplicar ninguna transformación de features (sin imputar faltantes, sin descartar columnas por correlación).

## Modelo registrado como Pipeline + pyfunc (en vez de `RandomForestClassifier` suelto)

Atiende el punto de la devolución sobre que el artefacto registrado en MLflow era solamente el clasificador, sin la imputación, el descarte de columnas ni el umbral de decisión que se aplicaban por fuera en el ETL — lo que hacía que el modelo registrado dependiera de que ese código externo no cambiara en el futuro.

En `entrenamiento_challenger.py`, el Pipeline de scikit-learn que se entrena ahora incluye como pasos la imputación de faltantes y el descarte de columnas por correlación (antes parte del viejo `proceso_etl`), además del `RandomForestClassifier`. Ese Pipeline se envuelve en un modelo `pyfunc` de MLflow (`ModeloConUmbral`) que además aplica el umbral de decisión de negocio sobre la probabilidad predicha, devolviendo directamente 0/1. Así, cada versión registrada en el Model Registry queda autosuficiente: recibe un dato crudo (todas las columnas originales) y devuelve la decisión final, sin pasos manuales aparte ni dependencia de un ETL externo que pueda cambiar de forma independiente. Como contrapartida, los datos que ahora consume este DAG desde `datos_splits/` (generados por `proceso_EL`) ya no vienen imputados ni filtrados por columnas, a diferencia de lo que entregaba el viejo `proceso_etl` bajo `datos_procesados/`.

## Transición del `champion` en el Model Registry

Al cambiar el contrato de entrada del modelo (ver entrada anterior), el `champion` que ya estaba registrado en MLflow quedó desactualizado: esperaba datos ya preprocesados (imputados y filtrados por columnas), no los datos crudos de 170 columnas que ahora produce `proceso_EL`. Al correr `entrenamiento_challenger` por primera vez con la arquitectura nueva, `comparar_y_promover` intentaba evaluar ese champion viejo con los datos nuevos y fallaba con un error de MLflow ("Failed to enforce schema of data"), porque los datos no cumplían el contrato con el que ese champion había sido registrado.

Se resolvió sacándole manualmente el alias `champion` a esa versión vieja desde la UI de MLflow. `comparar_y_promover` ya tenía previsto este caso: al no encontrar ningún modelo con el alias `champion`, entra en la rama de "arranque en frío" y registra y promueve directamente el challenger entrenado, sin comparar contra nada. Así, el primer modelo de la arquitectura nueva (Pipeline + pyfunc) quedó como `champion` vigente. Esto fue un paso manual de transición, necesario una sola vez por el cambio de contrato; de acá en adelante las comparaciones entre challenger y champion vuelven a ser automáticas.
