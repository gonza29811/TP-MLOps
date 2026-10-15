"""Prueba del endpoint /predict con un dataset sintético.

Genera registros manuales (valores bajos, valores extremos y valores nulos)
para validar que el pipeline de preprocesamiento embebido en el modelo
responda de forma robusta y sin fallar.
"""

import json
import urllib.request

API_URL = "http://localhost:8800/predict"

def main():
    # Dataset sintético: 3 escenarios críticos
    registros_sinteticos = [
        {"aa_000": 10, "ab_000": 0, "ac_000": 20},  # Escenario 1: Valores bajos (esperable: sano)
        {"aa_000": 800000, "ab_000": None, "ac_000": 2130706438},  # Escenario 2: Falla inminente + Nulos
        {}  # Escenario 3: Trama vacía (debería imputar todas las columnas con la mediana)
    ]

    pedido = urllib.request.Request(
        API_URL,
        data=json.dumps({"registros": registros_sinteticos}).encode(),
        headers={"Content-Type": "application/json"},
    )

    print("Enviando dataset sintético a la API de FastAPI...")
    try:
        respuesta = json.load(urllib.request.urlopen(pedido))
        print(f"\nModelo sirviendo: {respuesta['modelo']} v{respuesta['version']}")
        print(f"Umbral de decisión: {respuesta['umbral_decision']}\n")

        for i, pred in enumerate(respuesta["predicciones"]):
            print(f"Registro Sintético {i+1}:")
            print(f"  - Probabilidad de falla: {pred['probabilidad_falla']:.4f}")
            print(f"  - Decisión final: {'FALLA (1)' if pred['falla'] else 'SANO (0)'}\n")
            
    except urllib.error.URLError as e:
        print(f"Error: No se pudo conectar a FastAPI. ¿Está levantado el contenedor? Detalle: {e}")

if __name__ == "__main__":
    main()
