"""Servicio de predicción online del modelo champion de fallas del APS."""

import os
import threading

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
NOMBRE_MODELO = "modelo_scania_fallas"
ALIAS_CHAMPION = "champion"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(
    title="Servicio de modelo - Fallas del APS (Scania)",
    description=(
        "Predice si un camión tiene una falla del sistema de aire a presión (APS) "
        "usando el modelo `champion` vigente en el Model Registry de MLflow."
    ),
)

_lock = threading.Lock()
_cache = {"version": None, "pyfunc": None}


class PedidoPrediccion(BaseModel):
    """Uno o más registros de sensores; cada registro es {columna: valor}."""

    registros: list[dict[str, float | None]] = Field(
        ...,
        min_length=1,
        description=(
            "Lista de registros. Las columnas que falten o valgan null se tratan "
            "como valores faltantes (el modelo los imputa por la mediana)."
        ),
        examples=[[{"aa_000": 76698, "ab_000": None, "ac_000": 2130706438}]],
    )


def _cargar_champion():
    """Devuelve (version, modelo) del champion, recargando si cambió de versión."""
    cliente = MlflowClient()
    try:
        version = cliente.get_model_version_by_alias(
            NOMBRE_MODELO, ALIAS_CHAMPION
        ).version
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No hay un modelo con alias '{ALIAS_CHAMPION}' en el Model "
                "Registry todavía. Ejecutar el DAG entrenamiento_challenger."
            ),
        ) from exc

    with _lock:
        if _cache["version"] != version:
            _cache["pyfunc"] = mlflow.pyfunc.load_model(
                f"models:/{NOMBRE_MODELO}@{ALIAS_CHAMPION}"
            )
            _cache["version"] = version
        return _cache["version"], _cache["pyfunc"]


@app.get("/")
def read_root():
    return {"message": "Welcome to the Model Service"}


@app.get("/modelo")
def info_modelo():
    """Versión del champion que se está sirviendo y su umbral de decisión."""
    version, pyfunc = _cargar_champion()
    interno = pyfunc.unwrap_python_model()
    return {
        "modelo": NOMBRE_MODELO,
        "version": version,
        "umbral_decision": interno.umbral,
    }


@app.post("/predict")
def predecir(pedido: PedidoPrediccion):
    """Devuelve, por registro, la probabilidad de falla y la decisión final."""
    version, pyfunc = _cargar_champion()
    interno = pyfunc.unwrap_python_model()

    df = pd.DataFrame(pedido.registros).apply(pd.to_numeric, errors="coerce")
    df = df.astype("float64")

    # El pipeline espera todas las columnas de entrenamiento: las que no vengan
    # en el pedido quedan como NaN y las imputa el propio pipeline.
    columnas = getattr(interno.pipeline.steps[0][1], "columnas_", None)
    if columnas is not None:
        df = df.reindex(columns=columnas, fill_value=np.nan)

    probabilidades = interno.pipeline.predict_proba(df)[:, 1]
    return {
        "modelo": NOMBRE_MODELO,
        "version": version,
        "umbral_decision": interno.umbral,
        "predicciones": [
            {"probabilidad_falla": float(p), "falla": bool(p >= interno.umbral)}
            for p in probabilidades
        ],
    }
