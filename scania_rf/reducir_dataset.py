import shutil

ARCHIVOS = ["aps_failure_training_set.csv", "aps_failure_test_set.csv"]
LINEAS_METADATA_Y_HEADER = 21  # 20 lineas de licencia + 1 de header


def reducir_a_la_mitad(nombre_archivo):
    # Guarda una copia del dataset completo por si hay que volver atras
    shutil.copyfile(nombre_archivo, nombre_archivo.replace(".csv", "_completo.csv"))

    with open(nombre_archivo, "r", encoding="utf-8") as f:
        lineas = f.readlines()

    metadata_y_header = lineas[:LINEAS_METADATA_Y_HEADER]
    datos = lineas[LINEAS_METADATA_Y_HEADER:]
    datos_reducidos = datos[::2]  # se queda con 1 de cada 2 filas (mitad, distribuida)

    with open(nombre_archivo, "w", encoding="utf-8") as f:
        f.writelines(metadata_y_header)
        f.writelines(datos_reducidos)

    print(f"{nombre_archivo}: {len(datos)} filas -> {len(datos_reducidos)} filas")


for archivo in ARCHIVOS:
    reducir_a_la_mitad(archivo)