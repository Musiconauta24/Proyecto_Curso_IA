import numpy as np

class CModelo:
    def __init__(self, base_model, alpha=0.5):
        self.base_model = base_model
        self.alpha = alpha  # peso del efecto lluvia

    def predict(self, X, lluvia):
        """
        X: dataframe o array con columnas [Poblacion, Mes_Num, Temperatura_lag1]
        lluvia: array o serie con la lluvia del mes anterior
        """
        base_pred = self.base_model.predict(X)

        # Convertimos lluvia a numpy
        lluvia_arr = np.array(lluvia)

        # Ajuste con lluvia
        ajuste_lluvia = self.alpha * np.sqrt(np.maximum(0, lluvia_arr))
        pred_final = base_pred + ajuste_lluvia

        # Reemplazar NaN o Inf y devolver enteros
        pred_final = np.nan_to_num(pred_final, nan=0, posinf=0, neginf=0)
        return np.rint(pred_final).astype(int)
