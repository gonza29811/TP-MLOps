"""
Carga del dataset crudo al bucket de datos de MinIO (S3 local).

Este script es un "bootstrap" manual, no forma parte de ningun DAG: se
corre desde la maquina de cada integrante (no dentro de un contenedor de
Airflow) cada vez que hay datos nuevos para sumar al proyecto. Los deja
disponibles en el bucket `data` de MinIO, que es de donde despues el DAG
de ETL los va a leer (los contenedores de Airflow no tienen acceso al
filesystem local del repo, solo pueden leer del bucket).
"""

import os

import boto3
from dotenv import load_dotenv

# Cargamos el .env del repo (credenciales de MinIO y puertos de los
# servicios; load_dotenv busca hacia arriba si no lo encuentra en la
# carpeta actual).
load_dotenv()

MINIO_ENDPOINT_URL = f"http://localhost:{os.environ['MINIO_PORT']}"
DATA_BUCKET_NAME = os.environ["DATA_REPO_BUCKET_NAME"]

# Prefijo (carpeta logica dentro del bucket) donde se guardan los datos
# crudos, separados de los artefactos que despues genere el ETL.
PREFIJO_DATOS_ORIGINALES = "datos_originales"

# Archivos locales a subir: (ruta local, nombre de objeto en el bucket).
# Si en el futuro llegan datos nuevos con otro nombre de archivo, alcanza
# con agregarlos a esta lista.
ARCHIVOS_A_SUBIR = [
    ("aps_failure_training_set.csv", "aps_failure_training_set.csv"),
    ("aps_failure_test_set.csv", "aps_failure_test_set.csv"),
]


def crear_cliente_s3() -> boto3.client:
    """Crea un cliente de boto3 apuntando al MinIO local levantado por
    Docker Compose.

    :returns: cliente de S3 configurado contra MinIO.
    :rtype: boto3.client
    """
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT_URL,
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_ACCESS_KEY"],
    )


def asegurar_bucket(cliente_s3: boto3.client, nombre_bucket: str) -> None:
    """Crea el bucket si todavia no existe (no falla si ya esta creado).

    :param cliente_s3: cliente de S3 ya configurado.
    :type cliente_s3: boto3.client
    :param nombre_bucket: nombre del bucket a asegurar.
    :type nombre_bucket: str
    """
    buckets_existentes = {
        b["Name"] for b in cliente_s3.list_buckets()["Buckets"]
    }
    if nombre_bucket not in buckets_existentes:
        cliente_s3.create_bucket(Bucket=nombre_bucket)
        print(f"Bucket '{nombre_bucket}' creado.")


def subir_archivo(
    cliente_s3: boto3.client,
    nombre_bucket: str,
    ruta_local: str,
    nombre_objeto: str,
) -> None:
    """Sube un archivo local al bucket indicado, bajo el prefijo de datos
    crudos.

    :param cliente_s3: cliente de S3 ya configurado.
    :type cliente_s3: boto3.client
    :param nombre_bucket: bucket de destino.
    :type nombre_bucket: str
    :param ruta_local: ruta del archivo en el filesystem local.
    :type ruta_local: str
    :param nombre_objeto: nombre con el que va a quedar guardado dentro
        del bucket (sin el prefijo).
    :type nombre_objeto: str
    """
    clave_s3 = f"{PREFIJO_DATOS_ORIGINALES}/{nombre_objeto}"
    cliente_s3.upload_file(ruta_local, nombre_bucket, clave_s3)
    tamano_mb = os.path.getsize(ruta_local) / (1024 * 1024)
    print(f"Subido: {ruta_local} -> s3://{nombre_bucket}/{clave_s3} ({tamano_mb:.1f} MB)")


if __name__ == "__main__":
    s3 = crear_cliente_s3()
    asegurar_bucket(s3, DATA_BUCKET_NAME)

    for ruta_local, nombre_objeto in ARCHIVOS_A_SUBIR:
        subir_archivo(s3, DATA_BUCKET_NAME, ruta_local, nombre_objeto)

    print("\nListo. Los datos crudos ya estan disponibles en el bucket "
          f"'{DATA_BUCKET_NAME}' bajo el prefijo '{PREFIJO_DATOS_ORIGINALES}/'.")
