from flask import Flask, request, jsonify, render_template
import joblib, numpy as np, os, math
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modelo.modelo import CModelo  # 👈 Import necesario

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR, static_url_path="/static")

# Carga del modelo
modelo = joblib.load(os.path.join(BASE_DIR, "Modelo", "modelo_dengue_mejorado.pkl"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    required = ["Lluvia_mm_lag1", "Temperatura_lag1", "Poblacion", "Mes_Num"]
    for f in required:
        if f not in data:
            return jsonify({"error": f"Falta el campo requerido: {f}"}), 400

    # Convertir y validar
    try:
        lluvia = float(data["Lluvia_mm_lag1"])
        temperatura = float(data["Temperatura_lag1"])
        poblacion = float(data["Poblacion"])
        mes = int(data["Mes_Num"])
    except ValueError:
        return jsonify({"error": "Los valores deben ser numéricos"}), 400

    # 🚨 Validaciones de rango
    if not (50 <= lluvia <= 500):
        return jsonify({"error": "La lluvia debe estar entre 50 y 500 mm"}), 400

    if not (15 <= temperatura <= 40):
        return jsonify({"error": "La temperatura debe estar entre 15°C y 40°C"}), 400

    # Preparar entrada para el modelo
    entrada = np.array([[poblacion, mes, temperatura]])
    yhat = modelo.predict(entrada, [lluvia])[0]  # pasamos lluvia como lista
    yhat = abs(math.floor(yhat))  # entero y positivo

    return jsonify({"prediccion_casos_dengue": yhat})

if __name__ == '__main__':
    app.run(debug=True)
