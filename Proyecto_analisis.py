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

casos_dengue = dengue['FECHA REPORTE'].value_counts().reindex(anios_ordenados)
casos_dengue.plot(kind='bar', title='Número de personas contagiadas por año')
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


# ===============================
# Procesar DENGUE conservando Mes y Edad
# ===============================

# Normalizar columnas
dengue.columns = dengue.columns.str.strip()

# Mapeo de meses
meses_map = {
    'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
    'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
}

# ---- Preparar dataframe base ----
df_dengue = dengue[['FECHA REPORTE', 'MES REPORTE', 'MUNICIPIO REPORTE', 'EDAD']].copy()
df_dengue['MES_NUM'] = df_dengue['MES REPORTE'].map(meses_map)
df_dengue['Fecha'] = pd.to_datetime(dict(year=df_dengue['FECHA REPORTE'], month=df_dengue['MES_NUM'], day=1))

# Agrupar casos y edad (promedio)
df_dengue_group = (
    df_dengue.groupby(['MUNICIPIO REPORTE', 'Fecha'])
    .agg(
        Casos_Dengue=('EDAD', 'count'),   # Conteo de casos
        Edad=('EDAD', 'mean')             # Edad promedio en ese mes
    )
    .reset_index()
)

# Crear todas las combinaciones posibles de municipio y fecha
fechas_completas = pd.date_range(start='2018-01-01', end='2023-12-01', freq='MS')
municipios = dengue['MUNICIPIO REPORTE'].unique()
idx = pd.MultiIndex.from_product([municipios, fechas_completas], names=['MUNICIPIO REPORTE', 'Fecha'])

# Reindexar y rellenar datos
df_dengue_group = (
    df_dengue_group.set_index(['MUNICIPIO REPORTE', 'Fecha'])
    .reindex(idx)
    .reset_index()
)

# Rellenar Casos_Dengue con 0 donde no hay casos
df_dengue_group['Casos_Dengue'] = df_dengue_group['Casos_Dengue'].fillna(0).astype(int)

# Agregar columna "Mes" con nombre
df_dengue_group['Mes'] = df_dengue_group['Fecha'].dt.month.map({
    1: 'ENERO', 2: 'FEBRERO', 3: 'MARZO', 4: 'ABRIL', 5: 'MAYO', 6: 'JUNIO',
    7: 'JULIO', 8: 'AGOSTO', 9: 'SEPTIEMBRE', 10: 'OCTUBRE', 11: 'NOVIEMBRE', 12: 'DICIEMBRE'
})

print(df_dengue_group.head(15))


# =========================================
# Limpieza y verificación de los datos de lluvia
# # =========================================

# =========================================
# 1. Preparar fechas de referencia
# =========================================
fechas_completas = pd.date_range(start="2018-01-01", end="2023-12-01", freq="MS")

lluvia['Fecha'] = pd.to_datetime(lluvia['Fecha'], errors='coerce', dayfirst=True)
lluvia['Fecha'] = lluvia['Fecha'].dt.to_period('M').dt.to_timestamp()

# =========================================
# 2. Crear dataset de lluvia limpio por estación
# =========================================
lluvia_limpia = []

for estacion, df_estacion in lluvia.groupby("NombreEstacion"):
    df_estacion = (
        df_estacion.set_index("Fecha")
        .reindex(fechas_completas)
        .rename_axis("Fecha")
        .reset_index()
    )
    df_estacion["NombreEstacion"] = estacion
    
    # Reemplazar ceros por NaN
    df_estacion["Valor"] = df_estacion["Valor"].replace(0, np.nan)
    
    # Interpolación (sólo si no es MARACAIBO, se maneja después)
    if estacion != "MARACAIBO [4403000112]":
        df_estacion["Valor_interp"] = df_estacion["Valor"].interpolate(method="linear")
    else:
        df_estacion["Valor_interp"] = np.nan  # Se rellena más tarde con el promedio global
    
    lluvia_limpia.append(df_estacion)

lluvia_limpia = pd.concat(lluvia_limpia, ignore_index=True)

# =========================================
# 3. Calcular promedio global mensual (sin NaN ni ceros)
# =========================================
promedio_global = (
    lluvia_limpia.groupby("Fecha")["Valor_interp"]
    .mean()  
    .reset_index()
    .rename(columns={"Valor_interp": "PromedioGlobal"})
)

lluvia_limpia = lluvia_limpia.merge(promedio_global, on="Fecha", how="left")

# =========================================
# 4. Rellenar valores finales
# =========================================
# Caso especial: estación dañada "MARACAIBO [4403000112]" → siempre promedio global
lluvia_limpia.loc[
    lluvia_limpia["NombreEstacion"] == "MARACAIBO [4403000112]",
    "Valor_final"
] = lluvia_limpia.loc[
    lluvia_limpia["NombreEstacion"] == "MARACAIBO [4403000112]",
    "PromedioGlobal"
]

# Para las demás: usar interpolación primero, si aún falta, promedio global
lluvia_limpia["Valor_final"] = lluvia_limpia["Valor_interp"].fillna(lluvia_limpia["PromedioGlobal"])

# =========================================
# 5. Dataset final limpio
# =========================================
lluvia_final = lluvia_limpia[["NombreEstacion", "Fecha", "Valor_final"]].rename(columns={"Valor_final": "Lluvia_mm"})

# Verificación de los datos de lluvia
# Contar NaN totales en la columna Valor
total_nans = lluvia_final['Lluvia_mm'].isna().sum()
print(f"Total de NaN en 'Valor': {total_nans}")

# Contar NaN por estación
nans_por_estacion = lluvia_final[lluvia_final['Lluvia_mm'].isna()] \
    .groupby('NombreEstacion')['Lluvia_mm'] \
    .count()

print("\nNaN por Estación:")
print(nans_por_estacion)

#=============================================
# Revisar la distribución de la muestra de lluvia por año
#=============================================
# Revisar la distribución de la muestra de lluvia 
# --- 2️⃣ Agrupar Lluvia por año ---
lluvia_final['Fecha'] = pd.to_datetime(lluvia_final['Fecha'], errors='coerce', dayfirst=True)
lluvia_final['Año'] = lluvia_final['Fecha'].dt.year
# Calcular promedio de lluvia por año
promedio_lluvia = lluvia_final.groupby('Año')['Lluvia_mm'].mean()

# --- 3️⃣ Unir ambos DataFrames ---
df_combinado = pd.DataFrame({
    'Casos Dengue': casos_dengue,
    'Promedio Lluvia': promedio_lluvia
}).sort_index()

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel("Año")
ax1.set_ylabel("Casos Dengue", color="tab:orange")
ax1.bar(df_combinado.index - 0.2, df_combinado['Casos Dengue'], 
        width=0.4, color="tab:orange", label="Casos Dengue")
ax1.tick_params(axis='y', labelcolor="tab:orange")

# Eje Y derecho para promedio de lluvias
ax2 = ax1.twinx()
ax2.set_ylabel("Promedio Lluvia", color="tab:blue")
ax2.bar(df_combinado.index + 0.2, df_combinado['Promedio Lluvia'], 
        width=0.4, color="tab:blue", label="Promedio Lluvia")
ax2.tick_params(axis='y', labelcolor="tab:blue")

# Título y formato
plt.title("Casos de Dengue vs Promedio de Lluvia por Año")
fig.tight_layout()

# Mostrar
plt.show()

print ("\n Distribución de los casos por Mes")
# Ordenar los meses del año en español
meses_ordenados = ['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']

# --- 1️⃣ Casos de dengue por mes ---
casos_dengue_mes = dengue['MES REPORTE'].value_counts().reindex(meses_ordenados)

# --- 2️⃣ Lluvias por mes ---
# Asegurar formato fecha
lluvia['Fecha'] = pd.to_datetime(lluvia['Fecha'], errors='coerce', dayfirst=True)
# Extraer mes en texto para alinear con dengue
lluvia['Mes'] = lluvia['Fecha'].dt.month.apply(lambda x: meses_ordenados[x-1])
# Calcular promedio de lluvia por mes
promedio_lluvia_mes = lluvia.groupby('Mes')['Valor'].mean().reindex(meses_ordenados)

# --- 3️⃣ Unir ambos DataFrames ---
df_meses = pd.DataFrame({
    'Casos Dengue': casos_dengue_mes,
    'Promedio Lluvia': promedio_lluvia_mes
})

# --- 4️⃣ Graficar con doble eje Y ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# Casos de dengue (barras)
ax1.set_xlabel("Mes")
ax1.set_ylabel("Casos Dengue", color="tab:blue")
ax1.bar(df_meses.index, df_meses['Casos Dengue'], width=0.4, color="tab:blue", label="Casos Dengue")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax1.set_xticklabels(df_meses.index, rotation=45)

# Promedio de lluvias (línea con puntos)
ax2 = ax1.twinx()
ax2.set_ylabel("Promedio Lluvia", color="tab:orange")
ax2.plot(df_meses.index, df_meses['Promedio Lluvia'], color="tab:orange", marker='o', label="Promedio Lluvia")
ax2.tick_params(axis='y', labelcolor="tab:orange")

plt.title("Casos de Dengue vs Promedio de Lluvia por Mes")
fig.tight_layout()
plt.show()


# =========================================
# 1. Normalizar fechas en ambos DataFrames
# =========================================
lluvia_final['Fecha'] = pd.to_datetime(lluvia_final['Fecha'], errors='coerce', dayfirst=True)
lluvia_final['Fecha'] = lluvia_final['Fecha'].dt.to_period('M').dt.to_timestamp()  # primer día de mes

df_dengue_group['Fecha'] = pd.to_datetime(df_dengue_group['Fecha'])
df_dengue_group['Fecha'] = df_dengue_group['Fecha'].dt.to_period('M').dt.to_timestamp()

# =========================================
# 2. Diccionario de relación Municipio ↔ Estaciones
# =========================================
mapa_municipio_estaciones = {
    'SOLANO': ['ARARACUARA [44135010]', 'CUEMANI [44140020]', 'ESTRECHOS LOS [44127010]'],
    'BELEN DE LOS ANDAQUIES': ['BELEN DE ANDAQUIES [44040020]'],
    'SAN JOSE DEL FRAGUA': ['BELEN DE ANDAQUIES [44040020]'],
    'CARTAGENA DELCHAIRA': ['CARTAGENA D CHAIRA [46040010]', 'CUEMANI [44140020]'],
    'SAN VICENTE DEL CAGUAN': ['CARTAGENA D CHAIRA [46040010]'],
    'FLORENCIA': ['CORDOBA [44100010]', 'MACAGUAL [44035030]'],
    'MORELIA': ['MACAGUAL [44035030]'],
    'LA MONTANITA': ['LARANDIA [44030060]'],
    'MILAN': ['LARANDIA [44030060]'],
    'EL DONCELLO': ['MAGUARE - AUT [46035010]'],
    'EL PAUJIL': ['MAGUARE - AUT [46035010]'],
    'PUERTO RICO': ['MAGUARE - AUT [46035010]'],
    'ALBANIA': ['MARACAIBO [4403000112]'],
    'VALPARAISO': ['MARACAIBO [4403000112]'],
    'CURILLO': ['MARACAIBO [4403000112]'],
    'SOLITA': ['ESTRECHOS LOS [44127010]'],
}

# =========================================
# 3. Agrupar lluvia por estación y mes
# =========================================
lluvia_group = (
    lluvia_final.groupby(['NombreEstacion', 'Fecha'])['Lluvia_mm']
    .mean()
    .reset_index()
)

# =========================================
# 4. Calcular promedio de lluvia por municipio y mes
# =========================================
lista_resultados = []

for municipio, estaciones in mapa_municipio_estaciones.items():
    df_estaciones = lluvia_group[lluvia_group['NombreEstacion'].isin(estaciones)]
    
    # Promedio entre estaciones del municipio
    df_promedio = (
        df_estaciones.groupby('Fecha')['Lluvia_mm']
        .mean()
        .reset_index()
    )
    df_promedio['MUNICIPIO REPORTE'] = municipio
    lista_resultados.append(df_promedio)

# Unir todos los municipios en un solo DataFrame
df_lluvia_municipio = pd.concat(lista_resultados, ignore_index=True)

# =========================================
# 5. Unir con df_dengue_group
# =========================================
df_final = pd.merge(
    df_dengue_group,
    df_lluvia_municipio,
    how='left',
    left_on=['MUNICIPIO REPORTE', 'Fecha'],
    right_on=['MUNICIPIO REPORTE', 'Fecha']
)

df_final.rename(columns={'Valor': 'Lluvia_mm'}, inplace=True)

print(df_final.head(20))

#===========================================

# Contar NaN totales en la columna Lluvia_mm
total_nans = df_final['Lluvia_mm'].isna().sum()
print(f"Total de NaN en 'Lluvia_mm': {total_nans}")

# Contar NaN por municipio
nans_por_municipio = df_final[df_final['Lluvia_mm'].isna()] \
    .groupby('MUNICIPIO REPORTE')['Lluvia_mm'] \
    .count()

print("\nNaN por municipio:")
print(nans_por_municipio)

# Agregar la población de cada municipio para su análisis
# Diccionario de población adaptado a los nombres de df_final
poblacion_municipios = {
    "ALBANIA": 6432,
    "BELEN DE LOS ANDAQUIES": 11601,
    "CARTAGENA DELCHAIRA": 33908, 
    "CURILLO": 11737,
    "EL DONCELLO": 22183,
    "EL PAUJIL": 20528,
    "FLORENCIA": 175395,
    "LA MONTANITA": 23789,
    "MILAN": 11774,
    "MORELIA": 3836,
    "PUERTO RICO": 33447,
    "SAN JOSE DEL FRAGUA": 15029,
    "SAN VICENTE DEL CAGUAN": 69214,
    "SOLANO": 24131,
    "SOLITA": 9143,
    "VALPARAISO": 11687
}
# Agregar columna 'Poblacion' al df_final
df_final["Poblacion"] = df_final["MUNICIPIO REPORTE"].map(poblacion_municipios)

# Verificar: ver municipios únicos con su población
print(df_final[["MUNICIPIO REPORTE", "Poblacion"]].drop_duplicates())

# Para complementar la medición se agregan los datos de Temperatura

temp =  pd.read_csv('Temperatura.csv',sep=None, engine='python', encoding='latin-1', on_bad_lines='skip')
print(lluvia.head())

# Aseguramos que las fechas estén en formato datetime
temp['Fecha'] = pd.to_datetime(temp['Fecha'])
df_final['Fecha'] = pd.to_datetime(df_final['Fecha'])

# Extraer número de mes en ambos DataFrames
temp['Mes_Num'] = temp['Fecha'].dt.month
df_final['Mes_Num'] = df_final['Fecha'].dt.month

# Calcular promedio mensual en temp (agrupado por Mes_Num)
promedios_temp = (
    temp.groupby('Mes_Num')["Valor"]  # reemplaza "Valor" por el nombre real de la columna de temperatura en temp
    .mean()
    .reset_index()
    .rename(columns={"Valor": "Temperatura"})
)

# Hacer el merge con df_final
df_final = df_final.merge(promedios_temp, on='Mes_Num', how='left')

# Opcional: si ya no necesitas Mes_Num puedes borrarla
# df_final.drop(columns=['Mes_Num'], inplace=True)

print(df_final.head())

# Crear variables con rezago de 1 mes por municipio
df_final = df_final.sort_values(["MUNICIPIO REPORTE", "Fecha"])

df_final["Lluvia_mm_lag1"] = df_final.groupby("MUNICIPIO REPORTE")["Lluvia_mm"].shift(1)
df_final["Temperatura_lag1"] = df_final.groupby("MUNICIPIO REPORTE")["Temperatura"].shift(1)

# Matriz de correlación con los valores del mes anterior
correlacion_matriz = df_final[['Casos_Dengue', 'Lluvia_mm_lag1', 'Temperatura_lag1', 'Poblacion', 'Edad', 'Mes_Num']].corr()

# Graficar el mapa de calor
sns.heatmap(correlacion_matriz, annot=True, cmap='coolwarm')
plt.title('Correlación entre Casos de Dengue y condiciones ambientales del mes anterior')
plt.show()
#=======================================================
#guarda el dataset de los datos del dengue

df_final.to_csv("dengue_mensual.csv", index=False, encoding="utf-8")
print("✅ df_dengue_group guardado como 'dengue_mensual.csv'")