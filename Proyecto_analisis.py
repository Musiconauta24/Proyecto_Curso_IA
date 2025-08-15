import pandas as pd # pip install pandas
import matplotlib.pyplot as plt # pip install matplotlib
import seaborn as sns # pip install seaborn
import numpy as np


# Se Crean los dataset para trabajar las bases de datos
dengue = pd.read_csv('Casos_de_Dengue_Caqueta.csv')
lluvia = pd.read_csv('Precipitacion.csv',sep=None, engine='python', encoding='latin-1', on_bad_lines='skip')
# Visualización de los datos
print(dengue.head())
print(lluvia.head())

#===============================================================
# Se hace el proceso de anális y deporación de los datos de contagio de dengue

# Imprimir los datos nulos del sistema.
print(dengue.isnull().sum())
print(lluvia.isnull().sum())

#======================================================
#Analisis y depuración de los datos de Ciudad

# Con el análisis anterior deteminamos que la celda "Unnamed: 16",
# esta formada casi integramente de datos nulos por lo que la borramos
dengue.drop('Unnamed: 16',axis=1,inplace=True)

# Identificar inconsistencias en los nombres de ciudades
# ver una lista única de las ciudades ingresadas para detectar variaciones
municipios_unicos=dengue['MUNICIPIO REPORTE'].dropna().unique()

municipios_monitoreados = lluvia['NombreEstacion'].dropna().unique()

# Encontramos ahora que existen datos que no tienen definido el dato de
# que es un dato fundamental para la data por lo que tambien eliminamos las filas
# que tienen un dado de 'CAQUETA. MUNICIPIO DESCONOCIDO'ArithmeticError
# Eliminamos las filas donde 'MUNICIPIO REPORTE' es 'CAQUETA. MUNICIPIO DESCONOCIDO'
dengue = dengue[dengue['MUNICIPIO REPORTE'] != 'CAQUETA. MUNICIPIO DESCONOCIDO']
municipios_unicos=dengue['MUNICIPIO REPORTE'].dropna().unique()

# Revisar la distribución de la muestra
print ("\n Distribución de los casos por Municipio")
dengue['MUNICIPIO REPORTE'].value_counts().plot(kind='bar',title='Número de personas contagiadas por municipio')
plt.show()

#======================================================
#Análisis de los datos por Año y por mes del año.

print ("\n Distribución de los casos por Año")
# Obtener los años únicos y ordenarlos de menor a mayor
anios_ordenados = sorted(dengue['FECHA REPORTE'].unique())
# Graficar los casos por año en orden
dengue['FECHA REPORTE'].value_counts().reindex(anios_ordenados).plot(kind='bar', title='Número de personas contagiadas por año')
plt.show()

print ("\n Distribución de los casos por Mes")
# Ordenar los meses del año en español
meses_ordenados = ['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']
dengue['MES REPORTE'].value_counts().reindex(meses_ordenados).plot(kind='bar', title='Número de personas contagiadas por mes')
plt.show()

#======================================================
# Analisis y depuración de las edades de los contagiados

# Identificar edades inconsistentes
# consi# deremos que las edades válidas están entre 18 y 90
edades_inconsistentes=dengue[(dengue['EDAD']<1) | (dengue['EDAD']>100)|(dengue['EDAD'].isnull())]
print(edades_inconsistentes)

# Diagrama de distribución de edad de los contagiados con dengue
print ("Distribución de la muestra por rango de edad: ")
# Definir los rangos y etiquetas de edad
bins = [1, 18, 30, 45, 60, 100]
labels = ['1-17', '18-29', '30-44', '45-59', '60-100']

# Agrupar las edades en rangos
edades_rango = pd.cut(dengue['EDAD'], bins=bins, labels=labels, right=False)

# Mostrar el conteo de personas en cada grupo de edad
print(edades_rango.value_counts())

# Graficar la distribución por rango de edad
plt.figure()
edades_rango.value_counts().plot(kind="pie", labels=labels, title='Diagrama de torta de personas por rango de edad', autopct='%1.1f%%')
plt.ylabel('')
plt.show()


#==============================================================
# filtra la data de contagios por ciudad para cada mes del año


# Normalizar columnas
dengue.columns = dengue.columns.str.strip()

# Mapeo de meses
meses_map = {
    'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
    'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
}

# ---- Procesar DENGUE ----
df_dengue = dengue[['FECHA REPORTE', 'MES REPORTE', 'MUNICIPIO REPORTE']].copy()
df_dengue['MES_NUM'] = df_dengue['MES REPORTE'].map(meses_map)
df_dengue['Fecha'] = pd.to_datetime(dict(year=df_dengue['FECHA REPORTE'], month=df_dengue['MES_NUM'], day=1))

# Sumar casos
df_dengue_group = (
    df_dengue.groupby(['MUNICIPIO REPORTE', 'Fecha'])
    .size()
    .reset_index(name='Casos_Dengue')
)

# Crear todas las combinaciones posibles de municipio y fecha
fechas_completas = pd.date_range(start='2018-01-01', end='2023-12-01', freq='MS')
municipios = dengue['MUNICIPIO REPORTE'].unique()
idx = pd.MultiIndex.from_product([municipios, fechas_completas], names=['MUNICIPIO REPORTE', 'Fecha'])

# Reindexar el dataframe agrupado para incluir todas las fechas y municipios, rellenando con 0 donde no hay casos
df_dengue_group = df_dengue_group.set_index(['MUNICIPIO REPORTE', 'Fecha']).reindex(idx, fill_value=0).reset_index()


#Agragar una columna donde se incluyan las lluvias en ese mes para cada uno de los municipios

df_dengue_group['Lluvia_mm'] = np.random.uniform(0, 400, size=len(df_dengue_group))
print (df_dengue_group.head())

df_dengue_group.to_csv("df_dengue_group.csv", index=False, encoding="utf-8")
print("✅ df_dengue_group guardado como 'df_dengue_group.csv'")