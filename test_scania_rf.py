"""
Evaluacion del modelo de Random Forest entrenado para predecir fallas del
sistema de aire a presion (APS) en camiones Scania.

Consume el modelo y el umbral de decision generados por
train_scania_rf.py, junto con el set de test generado por etl_scania.py, y
genera las metricas y graficos de diagnostico finales.
"""

import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

COSTO_REVISION_INNECESARIA = 10  # costo de revisar un camion que no iba a fallar (FP)
COSTO_FALLA_NO_DETECTADA = 500  # costo de no detectar un camion que si falla (FN)


def calcular_costo(
    y_true,
    y_pred,
    costo_fp: float = COSTO_REVISION_INNECESARIA,
    costo_fn: float = COSTO_FALLA_NO_DETECTADA,
) -> int:
    """Calcula el costo de negocio total asociado a los errores de
    clasificacion.

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


def cargar_datos_test(carpeta: str = ".") -> tuple:
    """Carga el set de test generado por etl_scania.py.

    :param carpeta: carpeta donde se encuentran los CSV generados.
    :type carpeta: str
    :returns: tupla (X_test, y_test).
    :rtype: tuple
    """
    x_test = pd.read_csv(f"{carpeta}/X_test.csv")
    y_test = pd.read_csv(f"{carpeta}/y_test.csv").squeeze("columns")
    return x_test, y_test


def cargar_modelo_entrenado(
    ruta_modelo: str = "./modelo_rf.pkl", ruta_info: str = "./modelo_rf_info.json"
) -> tuple:
    """Carga el binario del modelo entrenado y su informacion asociada
    (hiperparametros y umbral de decision elegidos en train_scania_rf.py).

    :param ruta_modelo: ubicacion del binario del modelo.
    :type ruta_modelo: str
    :param ruta_info: ubicacion del JSON con la info del entrenamiento.
    :type ruta_info: str
    :returns: tupla (modelo, info_modelo).
    :rtype: tuple
    """
    with open(ruta_modelo, "rb") as f:
        modelo = pickle.load(f)
    with open(ruta_info) as f:
        info_modelo = json.load(f)
    return modelo, info_modelo


def calcular_metricas_clasificacion(y_true, y_pred) -> dict:
    """Calcula sensibilidad, especificidad, exactitud balanceada,
    precision, recall, F1 y costo de negocio para un modelo.

    :param y_true: valores reales.
    :param y_pred: valores predichos.
    :returns: diccionario con las metricas calculadas.
    :rtype: dict
    """
    verdaderos_neg, falsos_pos, falsos_neg, verdaderos_pos = confusion_matrix(
        y_true, y_pred
    ).ravel()
    sensibilidad = verdaderos_pos / (verdaderos_pos + falsos_neg)
    especificidad = verdaderos_neg / (verdaderos_neg + falsos_pos)

    return {
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "exactitud_balanceada": (sensibilidad + especificidad) / 2,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "costo": calcular_costo(y_true, y_pred),
        "falsos_positivos": int(falsos_pos),
        "falsos_negativos": int(falsos_neg),
    }


def graficar_matriz_confusion(y_true, y_pred, titulo: str, ruta_salida: str) -> None:
    """Grafica la matriz de confusion de un modelo y la guarda en disco.

    :param y_true: valores reales.
    :param y_pred: valores predichos.
    :param titulo: titulo del grafico.
    :type titulo: str
    :param ruta_salida: ubicacion donde guardar la imagen generada.
    :type ruta_salida: str
    """
    matriz_confusion = confusion_matrix(y_true, y_pred)

    _, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="cool", cbar=False, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    plt.tight_layout()
    plt.savefig(ruta_salida, bbox_inches="tight")
    plt.close()


def graficar_curva_roc(
    y_true, probabilidades, titulo: str, umbral_elegido: float, ruta_salida: str
) -> None:
    """Grafica la curva ROC de un modelo, marcando el punto de operacion
    correspondiente al umbral elegido, y la guarda en disco.

    :param y_true: valores reales.
    :param probabilidades: probabilidades predichas por el modelo.
    :param titulo: titulo del grafico.
    :type titulo: str
    :param umbral_elegido: umbral de decision usado en produccion.
    :type umbral_elegido: float
    :param ruta_salida: ubicacion donde guardar la imagen generada.
    :type ruta_salida: str
    """
    fpr, tpr, _ = roc_curve(y_true, probabilidades)
    area_bajo_curva = auc(fpr, tpr)

    prediccion = (probabilidades >= umbral_elegido).astype(int)
    verdaderos_neg, falsos_pos, falsos_neg, verdaderos_pos = confusion_matrix(
        y_true, prediccion
    ).ravel()
    fpr_operacion = falsos_pos / (falsos_pos + verdaderos_neg)
    tpr_operacion = verdaderos_pos / (verdaderos_pos + falsos_neg)

    _, ax = plt.subplots(figsize=(5, 5))
    etiqueta_modelo = f"{titulo} (AUC = {area_bajo_curva:.3f})"
    ax.plot(fpr, tpr, color="#ffa600", label=etiqueta_modelo)
    ax.plot(
        [0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
        label="Azar (AUC = 0.5)",
    )
    ax.scatter(
        [fpr_operacion], [tpr_operacion], color="#ffa600", edgecolor="k",
        zorder=5, s=60, label=f"Umbral elegido ({umbral_elegido:.2f})",
    )
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (Recall / TPR)")
    ax.set_title(f"Curva ROC - {titulo}")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(ruta_salida, bbox_inches="tight")
    plt.close()


def graficar_importancia_features(
    modelo, nombres_features, ruta_salida: str, n_top: int = 10
) -> None:
    """Grafica las n_top features mas importantes segun el modelo entrenado
    y la guarda en disco.

    :param modelo: modelo entrenado con atributo feature_importances_.
    :param nombres_features: nombres de las columnas de entrada del modelo.
    :param ruta_salida: ubicacion donde guardar la imagen generada.
    :type ruta_salida: str
    :param n_top: cantidad de features a graficar.
    :type n_top: int
    """
    importancias = modelo.feature_importances_
    orden = np.argsort(importancias)[-n_top:]

    _, ax = plt.subplots(figsize=(7, 6))
    nombres_top = np.array(nombres_features)[orden]
    ax.barh(nombres_top, importancias[orden], color="#ffa600", alpha=0.85)
    ax.set_xlabel("Importancia (reduccion de impureza)")
    ax.set_title(f"Top {n_top} features - {type(modelo).__name__}")
    plt.tight_layout()
    plt.savefig(ruta_salida, bbox_inches="tight")
    plt.close()


# Testeamos el modelo
X_test, y_test = cargar_datos_test(".")
modelo_rf, info_modelo_rf = cargar_modelo_entrenado()
umbral_final = info_modelo_rf["umbral_decision"]

probs_test = modelo_rf.predict_proba(X_test)[:, 1]
y_pred_test = (probs_test >= umbral_final).astype(int)

titulo_modelo = f"Random Forest (umbral {umbral_final:.2f})"
graficar_matriz_confusion(y_test, y_pred_test, titulo_modelo, "./matriz_confusion.png")
graficar_curva_roc(y_test, probs_test, "Random Forest", umbral_final, "./curva_roc.png")
graficar_importancia_features(modelo_rf, X_test.columns, "./importancia_features.png")

metricas_test = calcular_metricas_clasificacion(y_test, y_pred_test)

with open("./log_testeo.txt", "w") as f:
    f.write(f"Umbral de decision usado: {umbral_final:.2f}\n\n")
    f.write(f"Sensibilidad (Recall): {metricas_test['sensibilidad']:.4f}\n")
    f.write(f"Especificidad:         {metricas_test['especificidad']:.4f}\n")
    f.write(f"Exactitud balanceada:  {metricas_test['exactitud_balanceada']:.4f}\n")
    f.write(f"Precision:             {metricas_test['precision']:.4f}\n")
    f.write(f"Recall:                {metricas_test['recall']:.4f}\n")
    f.write(f"F1-score:              {metricas_test['f1']:.4f}\n")
    f.write(
        f"Costo total = {metricas_test['costo']} "
        f"(FP: {metricas_test['falsos_positivos']}, "
        f"FN: {metricas_test['falsos_negativos']})\n\n"
    )
    f.write(classification_report(y_test, y_pred_test, digits=3))
