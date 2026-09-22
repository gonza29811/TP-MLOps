"""
ETL del dataset de fallas del sistema de aire a presion (APS) de camiones
Scania: carga los datos crudos, limpia, imputa valores faltantes, filtra
features por correlacion y genera los artefactos (CSVs) que consumen los
scripts de entrenamiento y testeo.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Umbrales usados en la seleccion de features
UMBRAL_CORRELACION_ALTA = 0.9
UMBRAL_CORRELACION_BAJA = 0.075

COLUMNA_TARGET = "class"
COLUMNA_CONSTANTE = "cd_000"
MAPEO_CLASES = {"neg": 0, "pos": 1}

RANDOM_STATE = 42


def cargar_datos(ruta_train: str, ruta_test: str, filas_a_saltear: int = 20) -> tuple:
    """Carga los datasets de entrenamiento y test del sistema APS.

    :param ruta_train: ruta al csv de entrenamiento.
    :type ruta_train: str
    :param ruta_test: ruta al csv de test.
    :type ruta_test: str
    :param filas_a_saltear: filas de metadata a ignorar al inicio del csv.
    :type filas_a_saltear: int
    :returns: tupla (train, test) con los dataframes cargados.
    :rtype: tuple
    """
    train = pd.read_csv(ruta_train, skiprows=filas_a_saltear, na_values="na")
    test = pd.read_csv(ruta_test, skiprows=filas_a_saltear, na_values="na")
    return train, test


def eliminar_columna_constante(
    train: pd.DataFrame, test: pd.DataFrame, columna: str
) -> tuple:
    """Elimina una columna constante que no aporta informacion al modelo.

    :param train: dataframe de entrenamiento.
    :type train: pd.DataFrame
    :param test: dataframe de test.
    :type test: pd.DataFrame
    :param columna: nombre de la columna constante a eliminar.
    :type columna: str
    :returns: tupla (train, test) sin la columna indicada.
    :rtype: tuple
    """
    return train.drop(columns=[columna]), test.drop(columns=[columna])


def imputar_valores_faltantes(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columnas_feature: pd.Index,
    estrategia: str = "median",
) -> tuple:
    """Imputa los valores faltantes de las columnas indicadas.

    Se usa la mediana por defecto ya que es la medida que menos distorsiona
    la distribucion original en variables muy asimetricas.

    :param train: dataframe de entrenamiento.
    :type train: pd.DataFrame
    :param test: dataframe de test.
    :type test: pd.DataFrame
    :param columnas_feature: columnas sobre las que imputar.
    :type columnas_feature: pd.Index
    :param estrategia: estrategia de imputacion de SimpleImputer.
    :type estrategia: str
    :returns: tupla (train, test) imputados.
    :rtype: tuple
    """
    imputer = SimpleImputer(strategy=estrategia)
    train = train.copy()
    test = test.copy()
    train[columnas_feature] = imputer.fit_transform(train[columnas_feature])
    test[columnas_feature] = imputer.transform(test[columnas_feature])
    return train, test


def eliminar_features_correlacionadas(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columnas_feature: pd.Index,
    umbral: float = UMBRAL_CORRELACION_ALTA,
) -> tuple:
    """Elimina, de cada par de features con correlacion absoluta mayor al
    umbral, una de las dos columnas.

    :param train: dataframe de entrenamiento.
    :type train: pd.DataFrame
    :param test: dataframe de test.
    :type test: pd.DataFrame
    :param columnas_feature: columnas candidatas a evaluar.
    :type columnas_feature: pd.Index
    :param umbral: umbral de correlacion absoluta a partir del cual se
        considera redundante una feature.
    :type umbral: float
    :returns: tupla (columnas_eliminadas, train, test).
    :rtype: tuple
    """
    corr_matrix = train[columnas_feature].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    columnas_a_eliminar = [col for col in upper.columns if any(upper[col] > umbral)]
    train = train.drop(columns=columnas_a_eliminar)
    test = test.drop(columns=columnas_a_eliminar)
    return columnas_a_eliminar, train, test


def separar_features_y_target(
    df: pd.DataFrame,
    columna_target: str = COLUMNA_TARGET,
    mapeo_clases: dict | None = None,
) -> tuple:
    """Separa un dataframe en matriz de features y vector target, mapeando
    las clases a valores numericos (0/1).

    :param df: dataframe con features y target.
    :type df: pd.DataFrame
    :param columna_target: nombre de la columna target.
    :type columna_target: str
    :param mapeo_clases: diccionario de mapeo de clases a valores numericos.
    :type mapeo_clases: dict | None
    :returns: tupla (features, target).
    :rtype: tuple
    """
    features = df.drop(columns=columna_target)
    target = df[columna_target].map(mapeo_clases)
    return features, target


def eliminar_features_baja_correlacion(
    features_train_full: pd.DataFrame,
    target_train_full: pd.Series,
    datasets: dict[str, pd.DataFrame],
    umbral: float = UMBRAL_CORRELACION_BAJA,
) -> tuple:
    """Elimina las features cuya correlacion absoluta con el target es menor
    al umbral, para reducir la carga computacional de los modelos.

    :param features_train_full: matriz de features de entrenamiento completa.
    :type features_train_full: pd.DataFrame
    :param target_train_full: target de entrenamiento completo.
    :type target_train_full: pd.Series
    :param datasets: otros dataframes de features a los que aplicar el mismo
        filtro (por ejemplo {"train": X_train, "test": X_test}).
    :type datasets: dict[str, pd.DataFrame]
    :param umbral: umbral de correlacion absoluta minima con el target.
    :type umbral: float
    :returns: tupla (columnas_eliminadas, features_train_full, datasets_filtrados).
    :rtype: tuple
    """
    corr_con_target = features_train_full.corrwith(target_train_full)
    corr_ordenada = corr_con_target.sort_values(key=lambda s: s.abs(), ascending=False)
    columnas_baja_corr = corr_ordenada[corr_ordenada.abs() < umbral].index

    features_train_full = features_train_full.drop(columns=columnas_baja_corr)
    datasets_filtrados = {
        nombre: df.drop(columns=columnas_baja_corr) for nombre, df in datasets.items()
    }
    return columnas_baja_corr, features_train_full, datasets_filtrados


# Proceso de Extract, Transform and Load
train, test = cargar_datos(
    "aps_failure_training_set.csv", "aps_failure_test_set.csv"
)
train, test = eliminar_columna_constante(train, test, COLUMNA_CONSTANTE)

columnas_feature = train.columns.drop(COLUMNA_TARGET)
train, test = imputar_valores_faltantes(train, test, columnas_feature)

columnas_alta_corr, train, test = eliminar_features_correlacionadas(
    train, test, columnas_feature, UMBRAL_CORRELACION_ALTA
)

X_train_full, y_train_full = separar_features_y_target(
    train, COLUMNA_TARGET, MAPEO_CLASES
)
X_test, y_test = separar_features_y_target(test, COLUMNA_TARGET, MAPEO_CLASES)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    stratify=y_train_full,
    random_state=RANDOM_STATE,
)

resultado_filtro = eliminar_features_baja_correlacion(
    X_train_full,
    y_train_full,
    {"train": X_train, "test": X_test, "val": X_val},
    UMBRAL_CORRELACION_BAJA,
)
columnas_baja_corr, X_train_full, datasets_filtrados = resultado_filtro
X_train = datasets_filtrados["train"]
X_test = datasets_filtrados["test"]
X_val = datasets_filtrados["val"]

# Generamos los artefactos que van a consumir train_scania_rf.py y
# test_scania_rf.py
X_train.to_csv("./X_train.csv", index=False)
X_val.to_csv("./X_val.csv", index=False)
X_test.to_csv("./X_test.csv", index=False)
y_train.to_csv("./y_train.csv", index=False)
y_val.to_csv("./y_val.csv", index=False)
y_test.to_csv("./y_test.csv", index=False)

with open("./log_etl.txt", "w") as f:
    f.write(f"Shape train original: {train.shape}\n")
    f.write(
        f"Columnas eliminadas por alta correlacion "
        f"(> {UMBRAL_CORRELACION_ALTA}): {len(columnas_alta_corr)}\n"
    )
    f.write(f"{list(columnas_alta_corr)}\n")
    f.write(
        f"Columnas eliminadas por baja correlacion con el target "
        f"(< {UMBRAL_CORRELACION_BAJA}): {len(columnas_baja_corr)}\n"
    )
    f.write(f"{columnas_baja_corr.tolist()}\n")
    f.write(
        f"Shapes finales -> train: {X_train.shape}, val: {X_val.shape}, "
        f"test: {X_test.shape}\n"
    )
