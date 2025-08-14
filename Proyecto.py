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
plt.show())

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