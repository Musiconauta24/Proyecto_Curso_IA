from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import math
import pandas as pd

# Cargar el modelo entrenado
model = joblib.load("./Modelo/modelo_dengue_mejorado.pkl")  # ruta ajustada

# le decimos a Flask dónde está la carpeta templates
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Validar entradas
    required_fields = ["Lluvia_mm_lag1", "Temperatura_lag1", "Poblacion", "Mes_Num"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Falta el campo requerido: {field}"}), 400

    # Orden correcto según el entrenamiento
    columnas = ["Poblacion", "Mes_Num", "Lluvia_mm_lag1", "Temperatura_lag1"]

    # Crear DataFrame con orden correcto
    entrada = pd.DataFrame([{
        "Poblacion": float(data["Poblacion"]),
        "Mes_Num": int(data["Mes_Num"]),
        "Lluvia_mm_lag1": float(data["Lluvia_mm_lag1"]),
        "Temperatura_lag1": float(data["Temperatura_lag1"])
    }])[columnas]

    # Predicción
    prediccion = model.predict(entrada)[0]
    prediccion = abs(math.floor(prediccion))  # entero positivo redondeado hacia abajo

    return jsonify({
        "prediccion_casos_dengue": int(prediccion)
    })

if __name__ == '__main__':
    app.run(debug=True)
