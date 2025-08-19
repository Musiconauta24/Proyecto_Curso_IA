# train_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ===============================
# 1. Cargar datos
# ===============================
df_dengue_group = pd.read_csv("./dengue_mensual.csv")  

# ===============================
# 2. Selección de variables (features)
# ===============================
# Usamos todas las variables útiles:
# - Poblacion: tamaño de población municipal
# - Edad: edad promedio de casos reportados (si aplica)
# - Mes_Num: estacionalidad (ciclos anuales)
# - Lluvia_mm_lag1: lluvia del mes anterior
# - Temperatura_lag1: temperatura del mes anterior
# Target: Casos_Dengue

features = [
    'Poblacion', 
    'Edad', 
    'Mes_Num', 
    'Lluvia_mm_lag1', 
    'Temperatura_lag1'
]

# Asegurar que no hay NaN en las features
X = df_dengue_group[features].fillna(0)
y = df_dengue_group['Casos_Dengue']

# ===============================
# 3. División train/test
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 4. Entrenar modelo de regresión
# ===============================
model = LinearRegression()
model.fit(X_train, y_train)

# ===============================
# 5. Evaluación
# ===============================
y_pred = model.predict(X_test)

print("Resultados del modelo mejorado:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# ===============================
# 6. Guardar el modelo
# ===============================
joblib.dump(model, "Modelo/modelo_dengue_mejorado.pkl")
print("Modelo guardado como modelo_dengue_mejorado.pkl")
