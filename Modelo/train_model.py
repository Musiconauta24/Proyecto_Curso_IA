# ============================================
# 📚 Importación de librerías
# ============================================

import pandas as pd  # Manejo y análisis de datos en tablas (dataframes)
from sklearn.linear_model import LinearRegression  # Modelo de regresión lineal
from sklearn.model_selection import train_test_split  # División de datos en entrenamiento y prueba
from sklearn.metrics import mean_absolute_error, r2_score  # Métricas de evaluación del modelo
import joblib  # Guardar y cargar modelos entrenados
import sys, os  # Manejo de rutas y sistema operativo

# ============================================
# Ajuste de ruta para importar módulos propios
# ============================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modelo.modelo import CModelo  # Importa clase personalizada para el modelo híbrido

# ===============================
# 1. Cargar datos
# ===============================
df_dengue_group = pd.read_csv("./dengue_mensual.csv")  

# ===============================
# 2. Selección de variables
# ===============================
features = ['Poblacion', 'Mes_Num', 'Temperatura_lag1']  # Variables independientes (sin lluvia)
X = df_dengue_group[features].fillna(0)
y = df_dengue_group['Casos_Dengue']  # Variable dependiente (lo que queremos predecir)

# ===============================
# 3. División train/test
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # Separa 80% entrenamiento y 20% prueba
)

# ===============================
# 4. Entrenar modelo base
# ===============================
model = LinearRegression()  # Se crea el modelo de regresión lineal
model.fit(X_train, y_train)  # Se entrena con los datos

# ===============================
# 5. Modelo híbrido con lluvia
# ===============================
lluvia_train = df_dengue_group['Lluvia_mm_lag1'].iloc[X_train.index]  # Extrae lluvia del set de entrenamiento
lluvia_test = df_dengue_group['Lluvia_mm_lag1'].iloc[X_test.index]  # Extrae lluvia del set de prueba

modelo = CModelo(model, alpha=0.7)  # Modelo híbrido que combina regresión y lluvia (alpha = peso relativo)
y_pred = modelo.predict(X_test, lluvia_test)  # Predicción con el modelo híbrido

# ===============================
# Resultados de evaluación
# ===============================
print("Resultados del modelo híbrido:")
print("MAE:", mean_absolute_error(y_test, y_pred))  # Error absoluto medio
print("R²:", r2_score(y_test, y_pred))  # Coeficiente de determinación

# ===============================
# 6. Guardar modelo
# ===============================
joblib.dump(modelo, "Modelo/modelo_dengue_mejorado.pkl")  # Guardar modelo entrenado
print("Modelo guardado como modelo_dengue_mejorado.pkl")
