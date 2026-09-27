from datetime import datetime, timedelta

from airflow.decorators import dag, task

default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="proceso_etl",
    default_args=default_args,
    schedule="0 8 * * 1",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    dagrun_timeout=timedelta(minutes=15),
    tags=["TP MLOps"],
)
def proceso_etl():
    @task.virtualenv(
        task_id="obtener_dataset",
        requirements=["boto3>=1.34"],
        system_site_packages=False,
    )
    def get_data():
        """
        Verifica que los datos originales existan en MinIO (bucket `data`,
        prefijo `datos_originales/`), subidos previamente por
        cargar_datos_originales_s3.py, y devuelve la ubicacion para que la
        siguiente tarea los descargue por su cuenta.
        """
        import boto3

        BUCKET_NAME = "data"
        PREFIJO = "datos_originales"
        ARCHIVOS = {
            "train": "aps_failure_training_set.csv",
            "test": "aps_failure_test_set.csv",
        }

        s3 = boto3.client("s3")

        print("🔎 Verificando datos originales en MinIO...")
        for nombre, archivo in ARCHIVOS.items():
            cabecera = s3.head_object(Bucket=BUCKET_NAME, Key=f"{PREFIJO}/{archivo}")
            tamano_mb = cabecera["ContentLength"] / (1024 * 1024)
            print(f"   {nombre}: {archivo} ({tamano_mb:.1f} MB) OK")

        return {"bucket": BUCKET_NAME, "prefijo": PREFIJO, "archivos": ARCHIVOS}

    @task.virtualenv(
        task_id="procesar_etl",
        requirements=["boto3>=1.34", "pandas>=2.0", "numpy>=1.26", "scikit-learn>=1.3"],
        system_site_packages=False,
    )
    def procesar_etl(config_datos: dict):
        """
        Descarga los datos originales desde MinIO, aplica la limpieza y
        preparacion equivalente a etl_scania.py, y sube los splits
        resultantes (train/val/test) de vuelta al bucket, bajo el prefijo
        `datos_procesados/`.

        La seleccion de features (columnas descartadas por alta
        correlacion entre si o baja correlacion con el target) se calcula
        una unica vez y se persiste en MinIO
        (`datos_procesados/columnas_descartadas.json`). Corridas
        posteriores reutilizan ese archivo en lugar de recalcular los
        umbrales sobre los datos de turno: si no lo hicieramos, el
        esquema de columnas podria cambiar cada vez que cambia el
        dataset de entrada (por ejemplo al pasar de un dataset reducido a
        uno completo), dejando incompatible cualquier modelo ya
        entrenado con el esquema anterior.

        :param config_datos: bucket, prefijo y nombres de archivo de los
            datos originales (lo devuelve la tarea obtener_dataset).
        """
        import json

        import boto3
        import numpy as np
        import pandas as pd
        from botocore.exceptions import ClientError
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split

        UMBRAL_CORRELACION_ALTA = 0.9
        UMBRAL_CORRELACION_BAJA = 0.075
        COLUMNA_TARGET = "class"
        COLUMNA_CONSTANTE = "cd_000"
        MAPEO_CLASES = {"neg": 0, "pos": 1}
        RANDOM_STATE = 42

        bucket = config_datos["bucket"]
        prefijo_originales = config_datos["prefijo"]
        archivos = config_datos["archivos"]
        prefijo_procesados = "datos_procesados"
        clave_columnas_descartadas = f"{prefijo_procesados}/columnas_descartadas.json"

        s3 = boto3.client("s3")

        print("📥 Descargando datos originales desde MinIO...")
        ruta_train = "./aps_failure_training_set.csv"
        ruta_test = "./aps_failure_test_set.csv"
        s3.download_file(bucket, f"{prefijo_originales}/{archivos['train']}", ruta_train)
        s3.download_file(bucket, f"{prefijo_originales}/{archivos['test']}", ruta_test)

        train = pd.read_csv(ruta_train, skiprows=20, na_values="na")
        test = pd.read_csv(ruta_test, skiprows=20, na_values="na")

        print("🧹 Eliminando columna constante...")
        train = train.drop(columns=[COLUMNA_CONSTANTE])
        test = test.drop(columns=[COLUMNA_CONSTANTE])

        columnas_feature = train.columns.drop(COLUMNA_TARGET)

        print("🧹 Imputando valores faltantes (mediana)...")
        imputer = SimpleImputer(strategy="median")
        train[columnas_feature] = imputer.fit_transform(train[columnas_feature])
        test[columnas_feature] = imputer.transform(test[columnas_feature])

        print("Buscando esquema de columnas ya fijado en MinIO...")
        try:
            s3.download_file(bucket, clave_columnas_descartadas, "./columnas_descartadas.json")
            with open("./columnas_descartadas.json") as f:
                columnas_descartadas = json.load(f)
            columnas_alta_corr = columnas_descartadas["alta_correlacion"]
            columnas_baja_corr = columnas_descartadas["baja_correlacion"]
            print(
                f"   Esquema encontrado: se reutilizan {len(columnas_alta_corr)} + "
                f"{len(columnas_baja_corr)} columnas descartadas ya fijadas."
            )
        except ClientError as error:
            if error.response["Error"]["Code"] != "404":
                raise
            print("   No hay esquema fijado todavia: se calcula por primera vez.")

            print("Eliminando features con alta correlacion entre si...")
            corr_matrix = train[columnas_feature].corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            columnas_alta_corr = [
                col for col in upper.columns if any(upper[col] > UMBRAL_CORRELACION_ALTA)
            ]

            print("Eliminando features con baja correlacion con el target...")
            train_sin_alta_corr = train.drop(columns=columnas_alta_corr)
            X_train_full_temp = train_sin_alta_corr.drop(columns=COLUMNA_TARGET)
            y_train_full_temp = train_sin_alta_corr[COLUMNA_TARGET].map(MAPEO_CLASES)
            corr_con_target = X_train_full_temp.corrwith(y_train_full_temp)
            columnas_baja_corr = corr_con_target[
                corr_con_target.abs() < UMBRAL_CORRELACION_BAJA
            ].index.tolist()

            columnas_descartadas = {
                "alta_correlacion": columnas_alta_corr,
                "baja_correlacion": columnas_baja_corr,
            }
            with open("./columnas_descartadas.json", "w") as f:
                json.dump(columnas_descartadas, f, indent=2)
            s3.upload_file(
                "./columnas_descartadas.json", bucket, clave_columnas_descartadas
            )
            print(f"   Esquema calculado y fijado en s3://{bucket}/{clave_columnas_descartadas}")

        train = train.drop(columns=columnas_alta_corr)
        test = test.drop(columns=columnas_alta_corr)

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

        X_train = X_train.drop(columns=columnas_baja_corr)
        X_val = X_val.drop(columns=columnas_baja_corr)
        X_test = X_test.drop(columns=columnas_baja_corr)

        print("Subiendo splits procesados a MinIO...")
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
            s3.upload_file(ruta_local, bucket, f"{prefijo_procesados}/{nombre_archivo}")
            print(f"   {nombre_archivo} -> s3://{bucket}/{prefijo_procesados}/{nombre_archivo}")

        return {"bucket": bucket, "prefijo": prefijo_procesados}

    config_datos = get_data()
    procesar_etl(config_datos)


proceso_etl()
