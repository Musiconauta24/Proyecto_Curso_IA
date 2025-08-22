import pandas as pd # Manejo de datos tabulares
import matplotlib.pyplot as plt # Gráficas
import seaborn as sns # Mapas de calor y visualización avanzada
import numpy as np # Operaciones numéricas
from mpl_toolkits.mplot3d import Axes3D # Gráficas 3D
from sklearn.preprocessing import LabelEncoder # Codificación de etiquetas


# ==============================================================
# 1. CARGA DE DATOS
# ==============================================================

# Importar datasets principales
dengue = pd.read_csv('Casos_de_Dengue_Caqueta.csv')
lluvia = pd.read_csv('Precipitacion.csv', sep=None, engine='python', encoding='latin-1', on_bad_lines='skip')

# Vista rápida de los datos
print(dengue.head())
print(lluvia.head())


# ==============================================================
# 2. ANÁLISIS Y DEPURACIÓN DE LOS DATOS DE DENGUE Y LLUVIA
# ==============================================================

# Identificar valores nulos
print(dengue.isnull().sum())
print(lluvia.isnull().sum())


# --------------------------------------------------------------
# 2.1. Depuración de MUNICIPIOS
# --------------------------------------------------------------

# Revisar municipios únicos en dengue y estaciones de lluvia
municipios_unicos = dengue['MUNICIPIO REPORTE'].dropna().unique()
municipios_monitoreados = lluvia['NombreEstacion'].dropna().unique()

# Eliminar registros sin municipio válido
dengue = dengue[dengue['MUNICIPIO REPORTE'] != 'CAQUETA. MUNICIPIO DESCONOCIDO']
municipios_unicos = dengue['MUNICIPIO REPORTE'].dropna().unique()

# Visualizar distribución de casos por municipio
print("\nDistribución de los casos por Municipio")
dengue['MUNICIPIO REPORTE'].value_counts().plot(kind='bar', title='Número de contagios por municipio')
plt.show()


# --------------------------------------------------------------
# 2.2. Distribución por AÑO y MES
# --------------------------------------------------------------

print("\nDistribución de los casos por Año")
anios_ordenados = sorted(dengue['FECHA REPORTE'].unique())

# Casos de dengue por año
dengue['FECHA REPORTE'].value_counts().reindex(anios_ordenados).plot(kind='bar', title='Casos por año')
plt.show()

# Guardar serie de casos por año
casos_dengue = dengue['FECHA REPORTE'].value_counts().reindex(anios_ordenados)

# Casos por mes
print("\nDistribución de los casos por Mes")
meses_ordenados = ['ENERO','FEBRERO','MARZO','ABRIL','MAYO','JUNIO',
                   'JULIO','AGOSTO','SEPTIEMBRE','OCTUBRE','NOVIEMBRE','DICIEMBRE']
dengue['MES REPORTE'].value_counts().reindex(meses_ordenados).plot(kind='bar', title='Casos por mes')
plt.show()


# --------------------------------------------------------------
# 2.3. Depuración de EDADES
# --------------------------------------------------------------

# Identificar valores inconsistentes (<1 o >100)
edades_inconsistentes = dengue[(dengue['EDAD'] < 1) | (dengue['EDAD'] > 100) | (dengue['EDAD'].isnull())]
print(edades_inconsistentes)

# Agrupar en rangos de edad
print("Distribución por rangos de edad: ")
bins = [1, 18, 30, 45, 60, 100]
labels = ['1-17','18-29','30-44','45-59','60-100']
edades_rango = pd.cut(dengue['EDAD'], bins=bins, labels=labels, right=False)
print(edades_rango.value_counts())

# Gráfico de distribución por rango
plt.figure()
edades_rango.value_counts().plot(kind="pie", labels=labels, title='Casos por rango de edad', autopct='%1.1f%%')
plt.ylabel('')
plt.show()


# ==============================================================
# 3. PROCESAMIENTO DE LOS DATOS DE DENGUE (SERIE MENSUAL)
# ==============================================================

# Normalizar nombres de columnas
dengue.columns = dengue.columns.str.strip()

# Mapeo de meses a número
meses_map = {
    'ENERO': 1,'FEBRERO': 2,'MARZO': 3,'ABRIL': 4,'MAYO': 5,'JUNIO': 6,
    'JULIO': 7,'AGOSTO': 8,'SEPTIEMBRE': 9,'OCTUBRE': 10,'NOVIEMBRE': 11,'DICIEMBRE': 12
}

# Construcción del DataFrame mensual
df_dengue = dengue[['FECHA REPORTE','MES REPORTE','MUNICIPIO REPORTE','EDAD']].copy()
df_dengue['MES_NUM'] = df_dengue['MES REPORTE'].map(meses_map)
df_dengue['Fecha'] = pd.to_datetime(dict(year=df_dengue['FECHA REPORTE'], month=df_dengue['MES_NUM'], day=1))

# Crear todas las combinaciones municipio-fecha
fechas_completas = pd.date_range(start='2018-01-01', end='2023-12-01', freq='MS')
municipios = dengue['MUNICIPIO REPORTE'].unique()
idx = pd.MultiIndex.from_product([municipios, fechas_completas], names=['MUNICIPIO REPORTE', 'Fecha'])

# Agrupación de casos por municipio y mes
df_dengue_group = (
    df_dengue.groupby(['MUNICIPIO REPORTE','Fecha'])
    .agg(Casos_Dengue=('EDAD','count'))
    .reset_index()
)
df_dengue_group['Casos_Dengue'] = df_dengue_group['Casos_Dengue'].fillna(0).astype(int)

# Añadir nombre del mes
df_dengue_group['Mes'] = df_dengue_group['Fecha'].dt.month.map({i+1:m for i,m in enumerate(meses_ordenados)})

print(df_dengue_group.head(15))


# ==============================================================
# 4. LIMPIEZA Y PROCESAMIENTO DE DATOS DE LLUVIA
# ==============================================================

# 4.1 Normalizar fechas
fechas_completas = pd.date_range(start="2018-01-01", end="2023-12-01", freq="MS")
lluvia['Fecha'] = pd.to_datetime(lluvia['Fecha'], errors='coerce', dayfirst=True)
lluvia['Fecha'] = lluvia['Fecha'].dt.to_period('M').dt.to_timestamp()

# 4.2 Reconstrucción mensual por estación
lluvia_limpia = []
for estacion, df_estacion in lluvia.groupby("NombreEstacion"):
    df_estacion = (
        df_estacion.set_index("Fecha")
        .reindex(fechas_completas)
        .rename_axis("Fecha")
        .reset_index()
    )
    df_estacion["NombreEstacion"] = estacion
    
    # Eliminar valores fuera de rango (<30)
    df_estacion["Valor"] = df_estacion["Valor"].mask(df_estacion["Valor"] < 30, np.nan)
    
    # Interpolación excepto estación dañada
    if estacion != "MARACAIBO [4403000112]":
        df_estacion["Valor_interp"] = df_estacion["Valor"].interpolate(method="linear")
    else:
        df_estacion["Valor_interp"] = np.nan
    
    lluvia_limpia.append(df_estacion)

lluvia_limpia = pd.concat(lluvia_limpia, ignore_index=True)

# 4.3 Promedio global mensual
promedio_global = (
    lluvia_limpia.groupby("Fecha")["Valor_interp"]
    .mean()
    .reset_index()
    .rename(columns={"Valor_interp":"PromedioGlobal"})
)
lluvia_limpia = lluvia_limpia.merge(promedio_global, on="Fecha", how="left")

# 4.4 Relleno de valores finales
lluvia_limpia.loc[lluvia_limpia["NombreEstacion"]=="MARACAIBO [4403000112]","Valor_final"] = lluvia_limpia["PromedioGlobal"]
lluvia_limpia["Valor_final"] = lluvia_limpia["Valor_interp"].fillna(lluvia_limpia["PromedioGlobal"])

# 4.5 Dataset limpio final
lluvia_final = lluvia_limpia[["NombreEstacion","Fecha","Valor_final"]].rename(columns={"Valor_final":"Lluvia_mm"})

# Verificación de calidad
total_nans = lluvia_final['Lluvia_mm'].isna().sum()
print(f"Total de NaN en 'Valor': {total_nans}")
nans_por_estacion = lluvia_final[lluvia_final['Lluvia_mm'].isna()].groupby('NombreEstacion')['Lluvia_mm'].count()
print("\nNaN por estación:")
print(nans_por_estacion)


# ==============================================================
# 5. ANÁLISIS COMPARATIVO DENGUE vs LLUVIA
# ==============================================================

# 5.1 Promedio anual
lluvia_final['Año'] = lluvia_final['Fecha'].dt.year
promedio_lluvia = lluvia_final.groupby('Año')['Lluvia_mm'].mean()

# Combinar dengue y lluvia
df_combinado = pd.DataFrame({
    'Casos Dengue': casos_dengue,
    'Promedio Lluvia': promedio_lluvia
}).sort_index()

# Gráfico comparativo anual
fig, ax1 = plt.subplots(figsize=(10,6))
ax1.set_xlabel("Año")
ax1.set_ylabel("Casos Dengue", color="tab:orange")
ax1.bar(df_combinado.index-0.2, df_combinado['Casos Dengue'], width=0.4, color="tab:orange")
ax1.tick_params(axis='y', labelcolor="tab:orange")

ax2 = ax1.twinx()
ax2.set_ylabel("Promedio Lluvia", color="tab:blue")
ax2.bar(df_combinado.index+0.2, df_combinado['Promedio Lluvia'], width=0.4, color="tab:blue")
ax2.tick_params(axis='y', labelcolor="tab:blue")

plt.title("Casos de Dengue vs Promedio de Lluvia por Año")
fig.tight_layout()
plt.show()

# 5.2 Comparación mensual
print("\nDistribución de casos por Mes")
casos_dengue_mes = dengue['MES REPORTE'].value_counts().reindex(meses_ordenados)

lluvia['Fecha'] = pd.to_datetime(lluvia['Fecha'], errors='coerce', dayfirst=True)
lluvia['Mes'] = lluvia['Fecha'].dt.month.apply(lambda x: meses_ordenados[x-1])
promedio_lluvia_mes = lluvia.groupby('Mes')['Valor'].mean().reindex(meses_ordenados)

df_meses = pd.DataFrame({
    'Casos Dengue': casos_dengue_mes,
    'Promedio Lluvia': promedio_lluvia_mes
})

fig, ax1 = plt.subplots(figsize=(10,6))
ax1.set_xlabel("Mes")
ax1.set_ylabel("Casos Dengue", color="tab:blue")
ax1.bar(df_meses.index, df_meses['Casos Dengue'], width=0.4, color="tab:blue")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax1.set_xticklabels(df_meses.index, rotation=45)

ax2 = ax1.twinx()
ax2.set_ylabel("Promedio Lluvia", color="tab:orange")
ax2.plot(df_meses.index, df_meses['Promedio Lluvia'], color="tab:orange", marker='o')
ax2.tick_params(axis='y', labelcolor="tab:orange")

plt.title("Casos de Dengue vs Promedio de Lluvia por Mes")
fig.tight_layout()
plt.show()


# ==============================================================
# 6. INTEGRACIÓN MUNICIPIO ↔ ESTACIONES DE LLUVIA
# ==============================================================

# Normalizar fechas
lluvia_final['Fecha'] = pd.to_datetime(lluvia_final['Fecha'], errors='coerce', dayfirst=True).dt.to_period('M').dt.to_timestamp()
df_dengue_group['Fecha'] = pd.to_datetime(df_dengue_group['Fecha']).dt.to_period('M').dt.to_timestamp()

# Diccionario de correspondencia
mapa_municipio_estaciones = {
    'SOLANO':['ARARACUARA [44135010]','CUEMANI [44140020]','ESTRECHOS LOS [44127010]'],
    'BELEN DE LOS ANDAQUIES':['BELEN DE ANDAQUIES [44040020]'],
    'SAN JOSE DEL FRAGUA':['BELEN DE ANDAQUIES [44040020]'],
    'CARTAGENA DELCHAIRA':['CARTAGENA D CHAIRA [46040010]','CUEMANI [44140020]'],
    'SAN VICENTE DEL CAGUAN':['CARTAGENA D CHAIRA [46040010]'],
    'FLORENCIA':['CORDOBA [44100010]','MACAGUAL [44035030]'],
    'MORELIA':['MACAGUAL [44035030]'],
    'LA MONTANITA':['LARANDIA [44030060]'],
    'MILAN':['LARANDIA [44030060]'],
    'EL DONCELLO':['MAGUARE - AUT [46035010]'],
    'EL PAUJIL':['MAGUARE - AUT [46035010]'],
    'PUERTO RICO':['MAGUARE - AUT [46035010]'],
    'ALBANIA':['MARACAIBO [4403000112]'],
    'VALPARAISO':['MARACAIBO [4403000112]'],
    'CURILLO':['MARACAIBO [4403000112]'],
    'SOLITA':['ESTRECHOS LOS [44127010]']
}

# Agrupar lluvia por estación y mes
lluvia_group = lluvia_final.groupby(['NombreEstacion','Fecha'])['Lluvia_mm'].mean().reset_index()

# Calcular lluvia promedio por municipio
lista_resultados = []
for municipio, estaciones in mapa_municipio_estaciones.items():
    df_estaciones = lluvia_group[lluvia_group['NombreEstacion'].isin(estaciones)]
    df_promedio = df_estaciones.groupby('Fecha')['Lluvia_mm'].mean().reset_index()
    df_promedio['MUNICIPIO REPORTE'] = municipio
    lista_resultados.append(df_promedio)

df_lluvia_municipio = pd.concat(lista_resultados, ignore_index=True)

# Unir con casos de dengue
df_final = pd.merge(df_dengue_group, df_lluvia_municipio, how='left', on=['MUNICIPIO REPORTE','Fecha'])
df_final.rename(columns={'Valor':'Lluvia_mm'}, inplace=True)

print(df_final.head(20))


# ==============================================================
# 7. VARIABLES ADICIONALES: POBLACIÓN Y TEMPERATURA
# ==============================================================

# Agregar la población de cada municipio para su análisis
# Diccionario de población en miles para mejor predicción adaptado a los nombres de df_final
poblacion_municipios = {
    "ALBANIA": 6.432,
    "BELEN DE LOS ANDAQUIES": 11.601,
    "CARTAGENA DELCHAIRA": 33.908, 
    "CURILLO": 11.737,
    "EL DONCELLO": 22.183,
    "EL PAUJIL": 20.528,
    "FLORENCIA": 175.395,
    "LA MONTANITA": 23.789,
    "MILAN": 11.774,
    "MORELIA": 3.836,
    "PUERTO RICO": 33.447,
    "SAN JOSE DEL FRAGUA": 15.029,
    "SAN VICENTE DEL CAGUAN": 69.214,
    "SOLANO": 24.131,
    "SOLITA": 9.143,
    "VALPARAISO": 11.687
}
df_final["Poblacion"] = df_final["MUNICIPIO REPORTE"].map(poblacion_municipios)
print(df_final[["MUNICIPIO REPORTE","Poblacion"]].drop_duplicates())

# Incorporar datos de temperatura
temp = pd.read_csv('Temperatura.csv', sep=None, engine='python', encoding='latin-1', on_bad_lines='skip')
print(lluvia.head())

temp['Fecha'] = pd.to_datetime(temp['Fecha'])
df_final['Fecha'] = pd.to_datetime(df_final['Fecha'])

temp['Mes_Num'] = temp['Fecha'].dt.month
df_final['Mes_Num'] = df_final['Fecha'].dt.month

# Promedio mensual de temperatura
promedios_temp = (
    temp.groupby('Mes_Num')["Valor"]
    .mean()
    .reset_index()
    .rename(columns={"Valor":"Temperatura"})
)

df_final = df_final.merge(promedios_temp, on='Mes_Num', how='left')
print(df_final.head())


# ==============================================================
# 8. VARIABLES CON REZAGO (LAG DE 1 MES)
# ==============================================================

# Ordenar datos por municipio y fecha
df_final = df_final.sort_values(["MUNICIPIO REPORTE","Fecha"])

# Función para crear rezago circular
def lag_circular(x):
    shifted = x.shift(1)
    shifted.iloc[0] = x.iloc[-1]
    return shifted

# Crear variables rezagadas
df_final["Lluvia_mm_lag1"] = df_final.groupby("MUNICIPIO REPORTE")["Lluvia_mm"].transform(lag_circular)
df_final["Temperatura_lag1"] = df_final.groupby("MUNICIPIO REPORTE")["Temperatura"].transform(lag_circular)

# ===============================
# 9. Cluster de datos
# ===============================
# 1. Codificar municipios en números
le = LabelEncoder()
df_final["Municipio_Code"] = le.fit_transform(df_final["MUNICIPIO REPORTE"])
print (df_final)


# 2. Crear gráfico 3D
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Graficamos con color en función del municipio
sc = ax.scatter(
    df_final["Municipio_Code"],   # X = Municipio (codificado)
    df_final["Casos_Dengue"],     # Y = Casos de Dengue
    df_final["Lluvia_mm"],        # Z = Lluvia
    c=df_final["Municipio_Code"], # Color según municipio
    cmap="viridis",
    s=50
)

# 3. Etiquetas de los ejes
ax.set_xlabel("")
ax.set_ylabel("Casos de Dengue")
ax.set_zlabel("Lluvia (mm)")
plt.title("Relación Municipio - Casos de Dengue - Lluvia")

# 4. Agregar nombres reales de municipios en X
ax.set_xticks(df_final["Municipio_Code"].unique())
ax.set_xticklabels(le.inverse_transform(df_final["Municipio_Code"].unique()), rotation=45, ha="right")

# Mostrar gráfico
plt.show()

# Correlaciones
correlacion_matriz = df_final[['Casos_Dengue','Lluvia_mm_lag1','Temperatura_lag1','Poblacion','Mes_Num']].corr()
sns.heatmap(correlacion_matriz, annot=True, cmap='coolwarm')
plt.title('Correlación: Dengue vs Condiciones ambientales (rezago 1 mes)')
plt.show()


# ==============================================================
# 10. EXPORTACIÓN FINAL
# ==============================================================

df_final.to_csv("dengue_mensual.csv", index=False, encoding="utf-8")
print("✅ df_final guardado como 'dengue_mensual.csv'")
