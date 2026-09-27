from datetime import datetime, timedelta

from airflow.decorators import dag, task

default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="entrenamiento_challenger",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    dagrun_timeout=timedelta(minutes=30),
    tags=["TP MLOps"],
)
def entrenamiento_challenger():
    @task.virtualenv(
        task_id="entrenar_modelo",
        requirements=[
            "boto3>=1.34",
            "pandas>=2.0",
            "numpy>=1.26",
            "scikit-learn>=1.3",
            "matplotlib>=3.8",
            "mlflow>=3.16",
        ],
        system_site_packages=False,
    )
    def entrenar_modelo():
        """
        Descarga los datos procesados por proceso_etl desde MinIO, entrena
        un nuevo RandomForestClassifier (challenger) con busqueda de
        hiperparametros y eleccion de umbral de decision (igual criterio
        que train_scania_rf_mlflow.py), lo loguea como un run nuevo en
        MLflow -junto con la curva ROC y la matriz de confusion sobre el
        set de test como artefactos- y evalua su costo de negocio sobre
        ese mismo set.

        No lo registra todavia en el Model Registry: eso lo decide la
        siguiente tarea (comparar_y_promover), segun si supera o no al
        champion vigente.

        :returns: run_id de MLflow, path del artefacto del modelo dentro
            del run, umbral elegido y costo de negocio del challenger
            sobre el set de test.
        :rtype: dict
        """
        import boto3
        import mlflow
        import mlflow.sklearn
        import numpy as np
        import pandas as pd
        from mlflow.models import infer_signature
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import make_scorer
        from sklearn.model_selection import GridSearchCV

        BUCKET_NAME = "data"
        PREFIJO_DATOS_PROCESADOS = "datos_procesados"
        MLFLOW_TRACKING_URI = "http://mlflow:5000"
        MLFLOW_EXPERIMENT_NAME = "Fallas_Scania_APS"

        COSTO_REVISION_INNECESARIA = 10
        COSTO_FALLA_NO_DETECTADA = 500
        RANDOM_STATE = 42

        HIPERPARAMETROS_RF = {
            "n_estimators": [150, 200, 300],
            "max_depth": [5, 6, 8, 10],
            "min_samples_leaf": [10, 15, 20],
            "class_weight": ["balanced"],
        }
        UMBRALES_A_PROBAR = np.arange(0.41, 0.62, 0.01)

        def calcular_costo(y_true, y_pred):
            falsos_negativos = ((y_true == 1) & (y_pred == 0)).sum()
            falsos_positivos = ((y_true == 0) & (y_pred == 1)).sum()
            return int(
                falsos_negativos * COSTO_FALLA_NO_DETECTADA
                + falsos_positivos * COSTO_REVISION_INNECESARIA
            )

        print("Descargando datos procesados desde MinIO...")
        s3 = boto3.client("s3")
        for archivo in (
            "X_train.csv", "y_train.csv",
            "X_val.csv", "y_val.csv",
            "X_test.csv", "y_test.csv",
        ):
            s3.download_file(
                BUCKET_NAME, f"{PREFIJO_DATOS_PROCESADOS}/{archivo}", f"./{archivo}"
            )

        X_train = pd.read_csv("./X_train.csv")
        y_train = pd.read_csv("./y_train.csv").squeeze("columns")
        X_val = pd.read_csv("./X_val.csv")
        y_val = pd.read_csv("./y_val.csv").squeeze("columns")
        X_test = pd.read_csv("./X_test.csv")
        y_test = pd.read_csv("./y_test.csv").squeeze("columns")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        costo_scorer = make_scorer(calcular_costo, greater_is_better=False)

        print("Entrenando RandomForest (busqueda de hiperparametros)...")
        with mlflow.start_run(run_name="RandomForest_GridSearch_Airflow") as run:
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
                HIPERPARAMETROS_RF,
                refit=True,
                cv=5,
                scoring=costo_scorer,
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)
            mejor_modelo = grid_search.best_estimator_

            print("Buscando mejor umbral de decision sobre validacion...")
            probabilidades_val = mejor_modelo.predict_proba(X_val)[:, 1]
            mejor_umbral, mejor_costo_val = None, None
            for umbral in UMBRALES_A_PROBAR:
                prediccion = (probabilidades_val >= umbral).astype(int)
                costo = calcular_costo(y_val, prediccion)
                if mejor_costo_val is None or costo < mejor_costo_val:
                    mejor_costo_val, mejor_umbral = costo, float(umbral)

            prediccion_val = (probabilidades_val >= mejor_umbral).astype(int)
            falsos_negativos_val = int(((y_val == 1) & (prediccion_val == 0)).sum())
            falsos_positivos_val = int(((y_val == 0) & (prediccion_val == 1)).sum())

            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("umbral_decision", mejor_umbral)
            mlflow.log_metric("costo_negocio_val", float(mejor_costo_val))
            mlflow.log_metric("falsos_negativos_val", falsos_negativos_val)
            mlflow.log_metric("falsos_positivos_val", falsos_positivos_val)

            print("Evaluando al challenger sobre el set de test...")
            probabilidades_test = mejor_modelo.predict_proba(X_test)[:, 1]
            prediccion_test = (probabilidades_test >= mejor_umbral).astype(int)
            costo_test_challenger = calcular_costo(y_test, prediccion_test)
            falsos_negativos_test = int(((y_test == 1) & (prediccion_test == 0)).sum())
            falsos_positivos_test = int(((y_test == 0) & (prediccion_test == 1)).sum())
            mlflow.log_metric("costo_negocio_test", float(costo_test_challenger))
            mlflow.log_metric("falsos_negativos_test", falsos_negativos_test)
            mlflow.log_metric("falsos_positivos_test", falsos_positivos_test)

            print("Generando curva ROC...")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import (
                ConfusionMatrixDisplay,
                auc,
                confusion_matrix,
                roc_curve,
            )

            fpr, tpr, umbrales_roc = roc_curve(y_test, probabilidades_test)
            area_bajo_curva = auc(fpr, tpr)
            indice_umbral_elegido = (np.abs(umbrales_roc - mejor_umbral)).argmin()

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"ROC (AUC = {area_bajo_curva:.3f})")
            ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax_roc.scatter(
                fpr[indice_umbral_elegido],
                tpr[indice_umbral_elegido],
                color="red",
                zorder=5,
                label=f"Umbral elegido = {mejor_umbral:.2f}",
            )
            ax_roc.set_xlabel("Tasa de falsos positivos")
            ax_roc.set_ylabel("Tasa de verdaderos positivos")
            ax_roc.set_title("Curva ROC (set de test)")
            ax_roc.legend()
            mlflow.log_figure(fig_roc, "curva_roc.png")
            plt.close(fig_roc)

            print("Generando matriz de confusion...")
            matriz = confusion_matrix(y_test, prediccion_test)
            fig_matriz, ax_matriz = plt.subplots()
            ConfusionMatrixDisplay(
                confusion_matrix=matriz, display_labels=["Sin falla", "Con falla"]
            ).plot(ax=ax_matriz, cmap="Blues", colorbar=False)
            ax_matriz.set_title("Matriz de confusion (set de test)")
            mlflow.log_figure(fig_matriz, "matriz_confusion.png")
            plt.close(fig_matriz)

            signature = infer_signature(X_train, mejor_modelo.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=mejor_modelo,
                artifact_path="modelo_rf",
                signature=signature,
                serialization_format="cloudpickle",
            )
            run_id = run.info.run_id

        print(f"Challenger entrenado. Run: {run_id}. Costo test: {costo_test_challenger}")
        return {
            "run_id": run_id,
            "artifact_path": "modelo_rf",
            "umbral_decision": mejor_umbral,
            "costo_test_challenger": costo_test_challenger,
            "cantidad_muestras_test": len(y_test),
        }

    @task.virtualenv(
        task_id="comparar_y_promover",
        requirements=[
            "boto3>=1.34",
            "pandas>=2.0",
            "numpy>=1.26",
            "scikit-learn>=1.3",
            "mlflow>=3.16",
        ],
        system_site_packages=False,
    )
    def comparar_y_promover(resultado_challenger: dict):
        """
        Decide si el challenger recien entrenado reemplaza al champion
        vigente en el Model Registry de MLflow.

        - Si no hay ningun champion todavia (primera corrida): registra al
          challenger y le asigna el alias 'champion' directamente.
        - Si ya hay un champion: descarga su modelo y su umbral, lo evalua
          sobre el mismo set de test (descargado fresco desde MinIO) y
          compara el costo de negocio contra el del challenger, dividido
          por la cantidad de muestras de cada uno (para que la comparacion
          no se vea afectada si el tamano del set de test cambio entre
          una corrida y otra). Ese costo normalizado se usa solo para
          decidir, no se loguea como metrica.
            - Si el challenger gana: registra su version, mueve el alias
              'champion' vigente a 'champion_anterior' y asigna 'champion'
              a la version nueva.
            - Si el challenger pierde: no se registra ninguna version
              nueva. El run del challenger queda en el historial del
              experimento, sin alias.

        :param resultado_challenger: dict devuelto por entrenar_modelo.
        :type resultado_challenger: dict
        """
        import boto3
        import mlflow
        import mlflow.sklearn
        import pandas as pd
        from mlflow.exceptions import MlflowException
        from mlflow.tracking import MlflowClient

        BUCKET_NAME = "data"
        PREFIJO_DATOS_PROCESADOS = "datos_procesados"
        MLFLOW_TRACKING_URI = "http://mlflow:5000"
        NOMBRE_MODELO = "modelo_scania_fallas"
        ALIAS_CHAMPION = "champion"
        ALIAS_CHAMPION_ANTERIOR = "champion_anterior"

        def calcular_costo(y_true, y_pred, costo_fp=10, costo_fn=500):
            falsos_negativos = ((y_true == 1) & (y_pred == 0)).sum()
            falsos_positivos = ((y_true == 0) & (y_pred == 1)).sum()
            return int(falsos_negativos * costo_fn + falsos_positivos * costo_fp)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        run_id_challenger = resultado_challenger["run_id"]
        artifact_path_challenger = resultado_challenger["artifact_path"]
        costo_test_challenger = resultado_challenger["costo_test_challenger"]
        cantidad_muestras_challenger = resultado_challenger["cantidad_muestras_test"]

        def registrar_y_promover(version_anterior=None):
            model_uri = f"runs:/{run_id_challenger}/{artifact_path_challenger}"
            version_nueva = mlflow.register_model(model_uri, NOMBRE_MODELO)
            if version_anterior is not None:
                client.set_registered_model_alias(
                    NOMBRE_MODELO, ALIAS_CHAMPION_ANTERIOR, version_anterior
                )
            client.set_registered_model_alias(
                NOMBRE_MODELO, ALIAS_CHAMPION, version_nueva.version
            )
            print(f"Challenger promovido a champion (version {version_nueva.version}).")

        try:
            version_champion_actual = client.get_model_version_by_alias(
                NOMBRE_MODELO, ALIAS_CHAMPION
            )
        except MlflowException:
            version_champion_actual = None

        if version_champion_actual is None:
            print("No hay ningun champion todavia (arranque en frio).")
            registrar_y_promover(version_anterior=None)
            return

        print(f"Champion actual: version {version_champion_actual.version}. Evaluando...")
        umbral_champion = float(
            client.get_run(version_champion_actual.run_id).data.params["umbral_decision"]
        )
        modelo_champion = mlflow.sklearn.load_model(
            f"models:/{NOMBRE_MODELO}@{ALIAS_CHAMPION}"
        )

        s3 = boto3.client("s3")
        s3.download_file(BUCKET_NAME, f"{PREFIJO_DATOS_PROCESADOS}/X_test.csv", "./X_test.csv")
        s3.download_file(BUCKET_NAME, f"{PREFIJO_DATOS_PROCESADOS}/y_test.csv", "./y_test.csv")
        X_test = pd.read_csv("./X_test.csv")
        y_test = pd.read_csv("./y_test.csv").squeeze("columns")

        probabilidades_champion = modelo_champion.predict_proba(X_test)[:, 1]
        prediccion_champion = (probabilidades_champion >= umbral_champion).astype(int)
        costo_test_champion = calcular_costo(y_test, prediccion_champion)
        cantidad_muestras_champion = len(y_test)

        # Se normaliza por cantidad de muestras (costo promedio por fila) para
        # que la comparacion sea justa aunque el challenger y el champion se
        # hayan entrenado/evaluado con datasets de tamanos distintos. Este
        # valor normalizado se usa solo para decidir, no se loguea en MLflow.
        costo_por_muestra_challenger = costo_test_challenger / cantidad_muestras_challenger
        costo_por_muestra_champion = costo_test_champion / cantidad_muestras_champion

        print(
            f"Costo challenger: {costo_test_challenger} (n={cantidad_muestras_challenger}) | "
            f"Costo champion: {costo_test_champion} (n={cantidad_muestras_champion})"
        )

        if costo_por_muestra_challenger < costo_por_muestra_champion:
            print("El challenger gana. Se promueve.")
            registrar_y_promover(version_anterior=version_champion_actual.version)
        else:
            print("El challenger no supera al champion. No se promueve nada.")

    comparar_y_promover(entrenar_modelo())


entrenamiento_challenger()
