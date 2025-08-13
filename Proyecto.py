import pandas as pd # pip install pandas
import matplotlib.pyplot as plt # pip install matplotlib
import seaborn as sns # pip install seaborn

dengue = pd.read_csv('Casos_de_Dengue_Caqueta.csv')
lluvia = pd.read_csv('Precipitacion.csv',sep=None, engine='python', encoding='latin-1', on_bad_lines='skip')
# Visualización de los datos
print(dengue.head())
print(lluvia.head())

# Imprimir los datos nulos del sistema.
print(dengue.isnull().sum())
print(lluvia.isnull().sum())

# Con el análisis anterior deteminamos que la celda "Unnamed: 16",
# esta formada casi integramente de datos nulos por lo que la borramos
dengue.drop('Unnamed: 16',axis=1,inplace=True)

# Identificar edades inconsistentes
# consi# deremos que las edades válidas están entre 18 y 90
edades_inconsistentes=dengue[(dengue['EDAD']<1) | (dengue['EDAD']>100)|(dengue['EDAD'].isnull())]
print(edades_inconsistentes)