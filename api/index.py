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

    X_input = np.array([[
        float(data["Poblacion"]),
        int(data["Mes_Num"]),
        float(data["Temperatura_lag1"])
    ]])

    lluvia_input = [float(data["Lluvia_mm_lag1"])]

    yhat = modelo.predict(X_input, lluvia_input)[0]
    yhat = abs(math.floor(yhat))  # entero y positivo

    return jsonify({"prediccion_casos_dengue": yhat})

if __name__ == '__main__':
    app.run(debug=True)
