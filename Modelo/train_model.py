import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modelo.modelo import CModelo  # 👈 importamos la clase

# ===============================
# 1. Cargar datos
# ===============================
df_dengue_group = pd.read_csv("./dengue_mensual.csv")  

# ===============================
# 2. Selección de variables
# ===============================
features = ['Poblacion', 'Mes_Num', 'Temperatura_lag1']  # quitamos lluvia
X = df_dengue_group[features].fillna(0)
y = df_dengue_group['Casos_Dengue']

# ===============================
# 3. División train/test
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 4. Entrenar modelo base
# ===============================
model = LinearRegression()
model.fit(X_train, y_train)

# ===============================
# 5. Modelo híbrido con lluvia
# ===============================
lluvia_train = df_dengue_group['Lluvia_mm_lag1'].iloc[X_train.index]
lluvia_test = df_dengue_group['Lluvia_mm_lag1'].iloc[X_test.index]

modelo = CModelo(model, alpha=0.7)  # puedes ajustar alpha
y_pred = modelo.predict(X_test, lluvia_test)

print("Resultados del modelo híbrido:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# ===============================
# 6. Guardar modelo
# ===============================
joblib.dump(modelo, "Modelo/modelo_dengue_mejorado.pkl")
print("Modelo guardado como modelo_dengue_mejorado.pkl")
