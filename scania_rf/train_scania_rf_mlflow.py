"""
Entrenamiento de un modelo de Random Forest para predecir fallas del
sistema de aire a presion (APS) en camiones Scania, optimizando una
funcion de costo de negocio que penaliza mas los falsos negativos (fallas
no detectadas) que los falsos positivos (revisiones innecesarias).

Consume los artefactos generados por etl_scania.py (X_train, y_train,
X_val, y_val) y genera el binario del modelo entrenado junto con el
umbral de decision elegido.

Esta version agrega el registro del experimento y del modelo en MLflow,
ademas de mantener las salidas locales (pkl/json/txt) que ya consume
test_scania_rf.py.
"""

import json
import os
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

# Cargamos el .env del repo (credenciales de MinIO y puertos de MLflow, esta
# en la raiz del proyecto; load_dotenv busca hacia arriba si no lo encuentra
# en la carpeta actual) y lo traducimos a los nombres de variable que
# esperan boto3/MLflow.
load_dotenv()

os.environ["AWS_ACCESS_KEY_ID"] = os.environ["MINIO_ACCESS_KEY"]
os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["MINIO_SECRET_ACCESS_KEY"]
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://localhost:{os.environ['MINIO_PORT']}"

MLFLOW_TRACKING_URI = f"http://localhost:{os.environ['MLFLOW_PORT']}"
MLFLOW_EXPERIMENT_NAME = "Fallas_Scania_APS"
MLFLOW_MODEL_NAME = "modelo_scania_fallas"

# Costos de negocio (ver notebook original, seccion "Planteo del problema")
COSTO_REVISION_INNECESARIA = 10  # costo de revisar un camion que no iba a fallar (FP)
COSTO_FALLA_NO_DETECTADA = 500  # costo de no detectar un camion que si falla (FN)

RANDOM_STATE = 42

HIPERPARAMETROS_RF = {
    "n_estimators": [150, 200, 300],
    "max_depth": [5, 6, 8, 10],
    "min_samples_leaf": [10, 15, 20],
    "class_weight": ["balanced"],
}

UMBRALES_A_PROBAR = np.arange(0.41, 0.62, 0.01)


def calcular_costo(
    y_true,
    y_pred,
    costo_fp: float = COSTO_REVISION_INNECESARIA,
    costo_fn: float = COSTO_FALLA_NO_DETECTADA,
) -> int:
    """Calcula el costo de negocio total asociado a los errores de
    clasificacion.

    Costo Total = (costo_fp * cantidad de falsos positivos) +
    (costo_fn * cantidad de falsos negativos)

    :param y_true: valores reales (0 = sin falla, 1 = con falla).
    :param y_pred: valores predichos por el modelo.
    :param costo_fp: costo de un falso positivo (revision innecesaria).
    :type costo_fp: float
    :param costo_fn: costo de un falso negativo (falla no detectada).
    :type costo_fn: float
    :returns: costo total, como entero.
    :rtype: int
    """
    falsos_negativos = ((y_true == 1) & (y_pred == 0)).sum()
    falsos_positivos = ((y_true == 0) & (y_pred == 1)).sum()
    return int(falsos_negativos * costo_fn + falsos_positivos * costo_fp)


def cargar_datasets_entrenamiento(carpeta: str = ".") -> tuple:
    """Carga los artefactos de entrenamiento y validacion generados por
    etl_scania.py.

    :param carpeta: carpeta donde se encuentran los CSV generados.
    :type carpeta: str
    :returns: tupla (X_train, y_train, X_val, y_val).
    :rtype: tuple
    """
    x_train = pd.read_csv(f"{carpeta}/X_train.csv")
    y_train = pd.read_csv(f"{carpeta}/y_train.csv").squeeze("columns")
    x_val = pd.read_csv(f"{carpeta}/X_val.csv")
    y_val = pd.read_csv(f"{carpeta}/y_val.csv").squeeze("columns")
    return x_train, y_train, x_val, y_val


def entrenar_random_forest(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    grid_hiperparametros: dict,
    scorer,
    cv: int = 5,
) -> tuple:
    """Entrena un RandomForestClassifier con busqueda de hiperparametros
    por GridSearchCV.

    El tiempo de ejecucion y la memoria utilizada no se miden aca: MLflow
    ya registra automaticamente el inicio/fin (y por lo tanto la duracion)
    de cada run.

    :param features_train: matriz de features de entrenamiento.
    :type features_train: pd.DataFrame
    :param target_train: target de entrenamiento.
    :type target_train: pd.Series
    :param grid_hiperparametros: grilla de hiperparametros a probar.
    :type grid_hiperparametros: dict
    :param scorer: funcion de scoring de sklearn.
    :param cv: cantidad de folds de cross-validation.
    :type cv: int
    :returns: tupla (grid_search_entrenado, metricas_de_entrenamiento).
    :rtype: tuple
    """
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        grid_hiperparametros,
        refit=True,
        cv=cv,
        scoring=scorer,
        n_jobs=-1,
    )

    grid_search.fit(features_train, target_train)

    metricas_entrenamiento = {
        "mejores_hiperparametros": grid_search.best_params_,
    }
    return grid_search, metricas_entrenamiento


def buscar_mejor_umbral(
    modelo,
    features_val: pd.DataFrame,
    target_val: pd.Series,
    umbrales,
) -> pd.DataFrame:
    """Evalua el costo de negocio para distintos umbrales de probabilidad y
    devuelve una tabla con los resultados de cada uno.

    :param modelo: modelo de clasificacion ya entrenado.
    :param features_val: matriz de features de validacion.
    :type features_val: pd.DataFrame
    :param target_val: target de validacion.
    :type target_val: pd.Series
    :param umbrales: coleccion de umbrales de probabilidad a evaluar.
    :returns: tabla con el costo de negocio de cada umbral.
    :rtype: pd.DataFrame
    """
    probabilidades = modelo.predict_proba(features_val)[:, 1]

    resultados = []
    for umbral in umbrales:
        prediccion = (probabilidades >= umbral).astype(int)
        resultados.append(
            {
                "umbral": umbral,
                "FN": int(((target_val == 1) & (prediccion == 0)).sum()),
                "FP": int(((target_val == 0) & (prediccion == 1)).sum()),
                "costo": calcular_costo(target_val, prediccion),
            }
        )
    return pd.DataFrame(resultados)


def registrar_modelo_champion(nombre_modelo: str, run_id: str, artifact_path: str) -> None:
    """Registra la version del modelo recien entrenado en el Model Registry
    de MLflow y le asigna el alias 'champion'.

    Si es la primera vez que se registra este nombre de modelo, MLflow lo
    crea automaticamente.

    :param nombre_modelo: nombre del modelo registrado en MLflow.
    :type nombre_modelo: str
    :param run_id: id del run de MLflow donde se logueo el modelo.
    :type run_id: str
    :param artifact_path: carpeta de artefactos donde se logueo el modelo
        dentro del run (la misma que se le paso a mlflow.sklearn.log_model).
    :type artifact_path: str
    """
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    version_creada = mlflow.register_model(model_uri, nombre_modelo)
    client.set_registered_model_alias(nombre_modelo, "champion", version_creada.version)


if __name__ == "__main__":
    # 1. Configurar MLflow: servidor de tracking y experimento (si el
    # experimento no existe todavia, mlflow.set_experiment lo crea solo)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # 2. Cargar datos y preparar scorer
    X_train, y_train, X_val, y_val = cargar_datasets_entrenamiento(".")
    costo_scorer = make_scorer(calcular_costo, greater_is_better=False)

    # 3. Iniciar el tracking de MLflow
    with mlflow.start_run(run_name="RandomForest_GridSearch") as run:
        # Entrenamiento
        grid_rf, metricas_entrenamiento_rf = entrenar_random_forest(
            X_train, y_train, HIPERPARAMETROS_RF, costo_scorer, cv=5
        )
        best_rf = grid_rf.best_estimator_

        # Busqueda de umbral
        tabla_umbrales = buscar_mejor_umbral(best_rf, X_val, y_val, UMBRALES_A_PROBAR)
        mejor_fila = tabla_umbrales.loc[tabla_umbrales["costo"].idxmin()]
        umbral_final = float(mejor_fila["umbral"])

        # 4. Logueo en MLflow (parametros y metricas)
        mlflow.log_params(metricas_entrenamiento_rf["mejores_hiperparametros"])
        mlflow.log_param("umbral_decision", umbral_final)
        mlflow.log_metric("costo_negocio_val", float(mejor_fila["costo"]))
        mlflow.log_metric("falsos_negativos_val", float(mejor_fila["FN"]))
        mlflow.log_metric("falsos_positivos_val", float(mejor_fila["FP"]))

        # 5. Firma y guardado del artefacto (modelo) en MinIO
        signature = infer_signature(X_train, best_rf.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=best_rf,
            artifact_path="modelo_rf",
            signature=signature,
            serialization_format="cloudpickle",
        )

        # 6. Registro del modelo en el Model Registry, marcado como "champion"
        registrar_modelo_champion(MLFLOW_MODEL_NAME, run.info.run_id, "modelo_rf")

        # 7. Generacion de artefactos locales (se mantiene para test_scania_rf.py)
        with open("./modelo_rf.pkl", "wb") as f:
            pickle.dump(best_rf, f)

        info_modelo = {
            "mejores_hiperparametros": metricas_entrenamiento_rf["mejores_hiperparametros"],
            "umbral_decision": umbral_final,
        }
        with open("./modelo_rf_info.json", "w") as f:
            json.dump(info_modelo, f, indent=2)

        with open("./log_entrenamiento.txt", "w") as f:
            f.write("Mejores hiperparametros:\n")
            f.write(f"{metricas_entrenamiento_rf['mejores_hiperparametros']}\n")
            f.write(f"Umbral de decision elegido (validacion): {umbral_final:.2f}\n")
            f.write("\nBarrido de umbrales evaluados:\n")
            f.write(tabla_umbrales.to_string(index=False))
            f.write("\n")
