"""Prueba de punta a punta del endpoint /predict con datos reales de test.

Toma una muestra del set de test guardado en MinIO (con casos de falla y sin
falla), la envía a la API y compara las predicciones con el valor real.

Se ejecuta dentro del contenedor de FastAPI, que ya tiene acceso a MinIO:

    docker compose exec -T fastapi python - < dockerfiles/fastapi/probar_api.py
"""

import io
import json
import urllib.request

import boto3
import pandas as pd

API_URL = "http://localhost:8800/predict"
BUCKET = "data"
PREFIJO = "datos_splits"
N_POSITIVOS = 30
N_NEGATIVOS = 170


def leer_csv(s3, nombre):
    """Descarga un CSV del bucket y lo devuelve como DataFrame."""
    obj = s3.get_object(Bucket=BUCKET, Key=f"{PREFIJO}/{nombre}")
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def main():
    s3 = boto3.client("s3")
    X = leer_csv(s3, "X_test.csv")
    y = leer_csv(s3, "y_test.csv").squeeze("columns")

    idx = list(y[y == 1].sample(N_POSITIVOS, random_state=0).index) + list(
        y[y == 0].sample(N_NEGATIVOS, random_state=0).index
    )
    muestra, reales = X.loc[idx], y.loc[idx].tolist()

    # NaN -> null, para que sea JSON válido
    registros = json.loads(muestra.to_json(orient="records"))
    pedido = urllib.request.Request(
        API_URL,
        data=json.dumps({"registros": registros}).encode(),
        headers={"Content-Type": "application/json"},
    )
    respuesta = json.load(urllib.request.urlopen(pedido))

    predichos = [int(p["falla"]) for p in respuesta["predicciones"]]
    vp = sum(r == 1 and p == 1 for r, p in zip(reales, predichos))
    fn = sum(r == 1 and p == 0 for r, p in zip(reales, predichos))
    fp = sum(r == 0 and p == 1 for r, p in zip(reales, predichos))
    vn = sum(r == 0 and p == 0 for r, p in zip(reales, predichos))

    print(f"Modelo: {respuesta['modelo']} v{respuesta['version']}")
    print(f"Umbral de decisión: {respuesta['umbral_decision']}")
    print(f"Registros enviados: {len(registros)} ({N_POSITIVOS} con falla)")
    print(f"Fallas detectadas: {vp}  | Fallas no detectadas: {fn}")
    print(f"Falsas alarmas: {fp}      | Aciertos sin falla: {vn}")


if __name__ == "__main__":
    main()
