from datetime import datetime, timedelta

from airflow.decorators import dag, task

default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="proceso_EL",
    default_args=default_args,
    schedule="0 8 * * 1",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    dagrun_timeout=timedelta(minutes=15),
    tags=["TP MLOps"],
)
def proceso_el():
    @task.virtualenv(
        task_id="obtener_datos_crudos",
        requirements=["boto3>=1.34", "requests>=2.31"],
        system_site_packages=False,
    )
    def obtener_datos_crudos():
        """
        Se asegura de que los datos originales existan en MinIO (bucket
        `data`, prefijo `datos_originales/`).

        Si ya estan (por ejemplo porque alguien los subio a mano con
        cargar_datos_originales_s3.py, o porque una corrida anterior de
        este mismo DAG ya los descargo), se reutilizan tal cual. Si no
        estan, se descargan automaticamente desde el repositorio UCI
        (dataset "APS Failure at Scania Trucks") y se suben a MinIO, para
        que el flujo pueda arrancar solo desde un clon limpio del
        proyecto, sin pasos manuales previos.

        :returns: bucket, prefijo y nombres de archivo de los datos
            originales, para que la siguiente tarea los descargue.
        :rtype: dict
        """
        import io
        import zipfile

        import boto3
        import requests
        from botocore.exceptions import ClientError

        BUCKET_NAME = "data"
        PREFIJO = "datos_originales"
        ARCHIVOS = {
            "train": "aps_failure_training_set.csv",
            "test": "aps_failure_test_set.csv",
        }
        URL_UCI = (
            "https://archive.ics.uci.edu/static/public/421/"
            "aps+failure+at+scania+trucks.zip"
        )

        s3 = boto3.client("s3")

        print("🔎 Verificando datos originales en MinIO...")
        faltantes = []
        for nombre, archivo in ARCHIVOS.items():
            try:
                s3.head_object(Bucket=BUCKET_NAME, Key=f"{PREFIJO}/{archivo}")
                print(f"   {nombre}: {archivo} ya esta en MinIO, se reutiliza.")
            except ClientError as error:
                if error.response["Error"]["Code"] != "404":
                    raise
                faltantes.append((nombre, archivo))

        if faltantes:
            print(f"📥 Descargando dataset crudo desde UCI ({URL_UCI})...")
            respuesta = requests.get(URL_UCI, timeout=180)
            respuesta.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(respuesta.content)) as zip_file:
                for nombre, archivo in faltantes:
                    with zip_file.open(archivo) as origen:
                        contenido = origen.read()
                    ruta_local = f"./{archivo}"
                    with open(ruta_local, "wb") as destino:
                        destino.write(contenido)
                    s3.upload_file(ruta_local, BUCKET_NAME, f"{PREFIJO}/{archivo}")
                    print(
                        f"   {archivo} descargado de UCI y subido a "
                        f"s3://{BUCKET_NAME}/{PREFIJO}/{archivo}"
                    )

        print("✅ Datos originales disponibles:")
        for nombre, archivo in ARCHIVOS.items():
            cabecera = s3.head_object(Bucket=BUCKET_NAME, Key=f"{PREFIJO}/{archivo}")
            tamano_mb = cabecera["ContentLength"] / (1024 * 1024)
            print(f"   {nombre}: {archivo} ({tamano_mb:.1f} MB) OK")

        return {"bucket": BUCKET_NAME, "prefijo": PREFIJO, "archivos": ARCHIVOS}

    @task.virtualenv(
        task_id="dividir_y_subir_splits",
        requirements=["boto3>=1.34", "pandas>=2.0", "scikit-learn>=1.3"],
        system_site_packages=False,
    )
    def dividir_y_subir_splits(config_datos: dict):
        """
        Descarga los datos originales desde MinIO y los separa en splits
        train/val/test (80/20 estratificado sobre el train original para
        separar validacion), sin aplicar ninguna transformacion de
        features: nada de imputacion ni de descarte de columnas por
        correlacion. Esa logica ahora vive dentro del Pipeline de
        scikit-learn que se entrena y empaqueta en
        entrenamiento_challenger.py, para que el modelo registrado en
        MLflow sea autosuficiente.

        Lo unico que se hace aca es mapear la clase (`neg`/`pos` -> 0/1)
        y separar target de features, que no es una decision que dependa
        de los datos (a diferencia de la mediana de imputacion o los
        umbrales de correlacion) sino una convencion fija. Los splits
        resultantes se suben a MinIO bajo el prefijo `datos_splits/`.

        :param config_datos: bucket, prefijo y nombres de archivo de los
            datos originales (lo devuelve la tarea obtener_datos_crudos).
        """
        import boto3
        import pandas as pd
        from sklearn.model_selection import train_test_split

        COLUMNA_TARGET = "class"
        MAPEO_CLASES = {"neg": 0, "pos": 1}
        RANDOM_STATE = 42

        bucket = config_datos["bucket"]
        prefijo_originales = config_datos["prefijo"]
        archivos = config_datos["archivos"]
        prefijo_splits = "datos_splits"

        s3 = boto3.client("s3")

        print("Descargando datos originales desde MinIO...")
        ruta_train = "./aps_failure_training_set.csv"
        ruta_test = "./aps_failure_test_set.csv"
        s3.download_file(bucket, f"{prefijo_originales}/{archivos['train']}", ruta_train)
        s3.download_file(bucket, f"{prefijo_originales}/{archivos['test']}", ruta_test)

        train = pd.read_csv(ruta_train, skiprows=20, na_values="na")
        test = pd.read_csv(ruta_test, skiprows=20, na_values="na")

        X_train_full = train.drop(columns=COLUMNA_TARGET)
        y_train_full = train[COLUMNA_TARGET].map(MAPEO_CLASES)
        X_test = test.drop(columns=COLUMNA_TARGET)
        y_test = test[COLUMNA_TARGET].map(MAPEO_CLASES)

        print("Separando train/validacion (80/20, estratificado)...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.2,
            stratify=y_train_full,
            random_state=RANDOM_STATE,
        )

        print("Subiendo splits (crudos, sin transformar) a MinIO...")
        splits = {
            "X_train.csv": X_train,
            "X_val.csv": X_val,
            "X_test.csv": X_test,
            "y_train.csv": y_train,
            "y_val.csv": y_val,
            "y_test.csv": y_test,
        }
        for nombre_archivo, df in splits.items():
            ruta_local = f"./{nombre_archivo}"
            df.to_csv(ruta_local, index=False)
            s3.upload_file(ruta_local, bucket, f"{prefijo_splits}/{nombre_archivo}")
            print(f"   {nombre_archivo} -> s3://{bucket}/{prefijo_splits}/{nombre_archivo}")

        return {"bucket": bucket, "prefijo": prefijo_splits}

    config_datos = obtener_datos_crudos()
    dividir_y_subir_splits(config_datos)


proceso_el()
