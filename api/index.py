# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Cargar el modelo entrenado
model = joblib.load(".\Modelo\modelo_dengue.pkl")

# Inicializar Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "API de Predicción de Casos de Dengue"

@app.route('/predict', methods=['POST'])
def predict():
    # Recibir JSON con el dato de lluvia
    data = request.get_json()
    
    # Validar
    if "lluvia_mm" not in data:
        return jsonify({"error": "Debe enviar el valor de 'lluvia_mm'"}), 400
    
    lluvia_mm = float(data["lluvia_mm"])
    
    # Hacer predicción
    prediccion = model.predict(np.array([[lluvia_mm]]))
    
    return jsonify({
        "lluvia_mm": lluvia_mm,
        "prediccion_casos_dengue": round(prediccion[0], 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
