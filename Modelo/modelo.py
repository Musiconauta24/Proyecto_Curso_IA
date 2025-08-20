import numpy as np

class CModelo:
    def __init__(self, base_model, alpha=0.5):
        # Guardamos el modelo base entrenado (ej: LinearRegression de sklearn)
        self.base_model = base_model
        
        # Factor de ponderación que controla cuánto influye la lluvia en la predicción
        self.alpha = alpha  

    def predict(self, X, lluvia):
        """
        Método de predicción del modelo híbrido
        
        Parámetros:
        ----------
        X : DataFrame o array
            Contiene las variables de entrada [Poblacion, Mes_Num, Temperatura_lag1]
        lluvia : array o Serie
            Valores de precipitación (lag de 1 mes) para ajustar la predicción
        """

        # 1️⃣ Predicción inicial del modelo base (ej. regresión lineal)
        base_pred = self.base_model.predict(X)

        # 2️⃣ Convertimos el parámetro lluvia a un array de numpy
        lluvia_arr = np.array(lluvia)

        # 3️⃣ Calculamos el ajuste adicional debido a la lluvia
        #    - sqrt suaviza el impacto (no es lineal)
        #    - np.maximum(0, lluvia_arr) asegura que la lluvia no sea negativa
        ajuste_lluvia = self.alpha * np.sqrt(np.maximum(0, lluvia_arr))

        # 4️⃣ Sumamos la predicción base + el ajuste de lluvia
        pred_final = base_pred + ajuste_lluvia

        # 5️⃣ Reemplazamos posibles valores problemáticos:
        #    - NaN -> 0
        #    - +Inf -> 0
        #    - -Inf -> 0
        pred_final = np.nan_to_num(pred_final, nan=0, posinf=0, neginf=0)

        # 6️⃣ Redondeamos al entero más cercano y convertimos a int
        #    (porque el número de casos de dengue debe ser entero)
        return np.rint(pred_final).astype(int)
