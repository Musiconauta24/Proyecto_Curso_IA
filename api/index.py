# ===============================
# 1. Importación de librerías
# ===============================
from flask import Flask, request, jsonify, render_template  # Flask y utilidades web
import joblib, numpy as np, os, math                       # Modelos, arrays y utilidades matemáticas
import sys, os

# Agregamos la ruta base del proyecto al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importamos la clase personalizada del modelo híbrido
from Modelo.modelo import CModelo


# ===============================
# 2. Configuración de rutas base
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")   # Carpeta de templates HTML
STATIC_DIR = os.path.join(BASE_DIR, "static")         # Carpeta de archivos estáticos (CSS, JS, imágenes)


# ===============================
# 3. Inicialización de la aplicación Flask
# ===============================
app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,  # Indicamos dónde están las vistas HTML
    static_folder=STATIC_DIR,       # Indicamos dónde están los archivos estáticos
    static_url_path="/static"       # URL base para acceder a /static
)


# ===============================
# 4. Carga del modelo previamente entrenado
# ===============================
modelo = joblib.load(os.path.join(BASE_DIR, "Modelo", "modelo_dengue_mejorado.pkl"))


# ===============================
# 5. Rutas de la aplicación
# ===============================

# Ruta principal -> muestra la página web inicial
@app.route('/')
def home():
    return render_template("index.html")


# Ruta de predicción -> recibe datos vía POST y devuelve la predicción en JSON
@app.route('/predict', methods=['POST'])
def predict():
    # Obtenemos los datos en formato JSON desde la petición
    data = request.get_json()

    # Campos requeridos que deben estar presentes en la petición
    required = ["Lluvia_mm_lag1", "Temperatura_lag1", "Poblacion", "Mes_Num"]
    for f in required:
        if f not in data:
            return jsonify({"error": f"Falta el campo requerido: {f}"}), 400

    # Intentamos convertir los valores a numéricos
    try:
        lluvia = float(data["Lluvia_mm_lag1"])
        temperatura = float(data["Temperatura_lag1"])
        poblacion = float(data["Poblacion"])
        mes = int(data["Mes_Num"])
    except ValueError:
        return jsonify({"error": "Los valores deben ser numéricos"}), 400

    # ===============================
    # 6. Validaciones de rango
    # ===============================
    if not (50 <= lluvia <= 600):
        return jsonify({"error": "La Precipitación mensual debe estar entre 50 y 600 mm"}), 400

    if not (15 <= temperatura <= 40):
        return jsonify({"error": "La temperatura debe estar entre 15°C y 40°C"}), 400

    # ===============================
    # 7. Suma de modelo para todo el departamento
    # ===============================
    municipios_poblacion = {
        6.432, 11.601, 33.908, 11.737, 22.183, 20.528, 175.395, 23.789, 11.774, 3.836, 33.447, 15.029, 69.214, 24.131, 9.143, 11.687
    }

    if poblacion == 0:  # Caso especial: todo el departamento
        suma_total = 0
        for pob in municipios_poblacion:
            entrada = np.array([[float(pob), float(mes), float(temperatura)]], dtype=float)
            yhat_mun = modelo.predict(entrada, [lluvia])[0]
            yhat_mun = max(0, int(round(yhat_mun)))
            suma_total += yhat_mun
        return jsonify({"prediccion_casos_dengue": suma_total})
    else:

        # ===============================
        # 8. Preparar entrada para el modelo invividual por municipio
        # ===============================
        entrada = np.array([[poblacion, mes, temperatura]])  # Vector de entrada con las variables
        yhat = modelo.predict(entrada, [lluvia])[0]          # Llamada al modelo híbrido con lluvia
        yhat = max(0, int(round(yhat)))                      # asegura enteros y no permite negativos

        # Retornamos la predicción en formato JSON
        return jsonify({"prediccion_casos_dengue": yhat})


# ===============================
# 9. Ejecutar servidor
# ===============================
if __name__ == '__main__':
    app.run(debug=True)
