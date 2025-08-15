# train_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Cargar los datos (simulados)
df_dengue_group = pd.read_csv("./df_dengue_group.csv")  

x=df_dengue_group[['Edad','GastoMensual','VisitasPorMes']]

# Variables de entrada (features) y salida (target)
X = df_dengue_group[['Lluvia_mm']]  # Predictor
y = df_dengue_group['Casos_Dengue']  # Variable a predecir

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluación rápida
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Guardar el modelo
joblib.dump(model, "Modelo\modelo_dengue.pkl")
print("Modelo guardado como modelo_dengue.pkl")
