##### GRAFICAS ################
# Importar modulos
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import seaborn as sns
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point
datos = pd.read_csv('D:/MASTER/Chicago/datos2.csv')
datos.info()
datos=datos.drop(datos.columns[0], axis=1)
# print(datos.head())
print("DELITOS POR TIPOS")
print("*****************")
print(datos['Primary Type'].value_counts())
print(datos['Arrest'].value_counts())
print(datos['Delito Grave'].value_counts())
# Se elige el año 2012 y se realizan graficas sobre datos socieconomicos en Chicago
# Se establece Fecha como indice
datos['Fecha'] = pd.to_datetime(datos.Fecha,format = '%d/%m/%Y')
datos.set_index('Fecha',inplace=True)
chi2012=datos.query("Year==2012")
############################################
##### DISTRIBUCIONES TEMPORALES ###########
# Se usa resample para transformar en frecuencias las series temporales
# Años
delitos_año=datos.resample("Y").count()
delitos_año=pd.DataFrame(delitos_año.iloc[:,0])
delitos_año.columns=["Delitos Totales / Año"]
delitos_año.plot()
plt.xlabel('Año')
plt.ylabel('Delitos Totales')
plt.title('Delitos por año')
# plt.axhline(delitos_mes['Delitos Totales / Mes'].mean(), color = 'red')
fig=plt.gcf()
fig.set_size_inches(10,5)

# Meses
delitos_mes=datos.resample("M").count()
delitos_mes=pd.DataFrame(delitos_mes.iloc[:,0])
delitos_mes.columns=["Delitos Totales / Mes"]
delitos_mes.plot()
plt.xlabel('Año')
plt.ylabel('Delitos Totales')
plt.title('Delitos por mes')
plt.axhline(delitos_mes['Delitos Totales / Mes'].mean(), color = 'red')
fig=plt.gcf()
fig.set_size_inches(10,5)


fig, ax = plt.subplots()
plt.xlabel('Meses')
plt.ylabel('Delitos Totales')
plt.title('Actividad Delictiva')
plt.axhline(delitos_mes['Delitos Totales / Mes'].mean(), color = 'red')
plt.xlabel('Meses')
leyenda = ["Enero",'Enero','Marzo','Mayo','Julio','Septiembre','Noviembre']
ax.set_xticklabels(leyenda)
fig=plt.gcf()
fig.set_size_inches(15,5)
ax.plot(delitos_mes['2008'].values, color = 'Blue', label = '2008')
ax.plot(delitos_mes['2009'].values, color = 'Green', label = '2009')
ax.plot(delitos_mes['2010'].values, color = 'Brown', label = '2010')
ax.plot(delitos_mes['2011'].values, color = 'Yellow', label = '2011')
ax.plot(delitos_mes['2012'].values, color = 'Magenta', label = '2012')
plt.legend()

# Dias

delitos_dia=datos.resample("d").count()
delitos_dia=pd.DataFrame(delitos_dia.iloc[:,0])
delitos_dia.columns=["Delitos Totales / Dia"]

delitos_dia.plot()
plt.xlabel('Año')
plt.ylabel('Delitos Totales')
plt.title('Delitos por dia')
plt.axhline(delitos_dia['Delitos Totales / Dia'].mean(), color = 'red')
fig=plt.gcf()
fig.set_size_inches(20,10)

max_dia=delitos_dia["Delitos Totales / Dia"].max()
print('El dia con mas delitos fue:')
delitos_dia[(delitos_dia['Delitos Totales / Dia'] == max_dia)]

min_dia=delitos_dia["Delitos Totales / Dia"].min()
print('El dia con menos delitos fue:')
delitos_dia[(delitos_dia['Delitos Totales / Dia'] == min_dia)]

media_dia=round(delitos_dia["Delitos Totales / Dia"].mean(),2)
print('La media de delitos por dia es :', media_dia)

# Horas
delitos_hora=[]
delitos_hora = datos.groupby(['Hora']).size()
delitos_hora=pd.DataFrame(delitos_hora)
delitos_hora.rename(columns={0:"Horas"}, inplace=True)
delitos_hora["Media_Hora"]=delitos_hora["Horas"]/(365*5)
x = delitos_hora.index
y = delitos_hora.Media_Hora
a = np.arange(0,24)
grid = sns.pointplot(x = x, y = y, data = delitos_hora)
grid.set_xticklabels(a)
plt.axhline(delitos_hora.Media_Hora.mean(), color = 'red')
fig = plt.gcf()
fig.set_size_inches(20,10)
plt.grid()
media_hora=round(delitos_hora.Media_Hora.mean(),2)
print('La media de delitos por hora es :', media_hora)
print("Horas mas activas")
print(delitos_hora.sort_values("Media_Hora", ascending=False).head(5))
print("Horas menos activas")
print(delitos_hora.sort_values("Media_Hora", ascending=False).tail(5))

# Mapas sobre la socieconomia
fig, axes = plt.subplots(2, 3, figsize=(15, 15)) 
fig.suptitle('Valores socieconomicos de Chicago', fontsize=20)
sns.scatterplot(ax=axes[0, 0], data=chi2012, x=chi2012['Longitude'], y=chi2012['Latitude'], hue=chi2012['PERCENT OF HOUSING CROWDED']) 
axes[0, 0].set_xlabel('Longitud', fontsize=12) 
axes[0, 0].set_ylabel('Latitud', fontsize=12) 
axes[0, 0].set_xlim([-87.95,-87.5]) # centro x del grafico
axes[0, 0].set_ylim([41.62,42.05]) # centro y del grafico
sns.scatterplot(ax=axes[0, 1], data=chi2012, x=chi2012['Longitude'], y=chi2012['Latitude'], hue=chi2012['PERCENT HOUSEHOLDS BELOW POVERTY']) 
axes[0, 1].set_xlabel('Longitud', fontsize=12) 
axes[0, 1].set_ylabel('Latitud', fontsize=12) 
axes[0, 1].set_xlim([-87.95,-87.5]) # centro x del grafico
axes[0, 1].set_ylim([41.62,42.05]) # centro y del grafico
sns.scatterplot(ax=axes[0, 2], data=chi2012, x=chi2012['Longitude'], y=chi2012['Latitude'], hue=chi2012['PERCENT AGED 16+ UNEMPLOYED']) 
axes[0, 2].set_xlabel('Longitud', fontsize=12) 
axes[0, 2].set_ylabel('Latitud', fontsize=12) 
axes[0, 2].set_xlim([-87.95,-87.5]) # centro x del grafico
axes[0, 2].set_ylim([41.62,42.05]) # centro y del grafico
sns.scatterplot(ax=axes[1, 0], data=chi2012, x=chi2012['Longitude'], y=chi2012['Latitude'], hue=chi2012['PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA']) 
axes[1, 0].set_xlabel('Longitud', fontsize=12) 
axes[1, 0].set_ylabel('Latitud', fontsize=12) 
axes[1, 0].set_xlim([-87.95,-87.5]) # centro x del grafico
axes[1, 0].set_ylim([41.62,42.05]) # centro y del grafico
sns.scatterplot(ax=axes[1, 1], data=chi2012, x=chi2012['Longitude'], y=chi2012['Latitude'], hue=chi2012['PERCENT AGED UNDER 18 OR OVER 64']) 
axes[1, 1].set_xlabel('Longitud', fontsize=12) 
axes[1, 1].set_ylabel('Latitud', fontsize=12) 
axes[1, 1].set_xlim([-87.95,-87.5]) # centro x del grafico
axes[1, 1].set_ylim([41.62,42.05]) # centro y del grafico
sns.scatterplot(ax=axes[1, 2], data=chi2012, x=chi2012['Longitude'], y=chi2012['Latitude'], hue=chi2012['PER CAPITA INCOME ']) 
axes[1, 2].set_xlabel('Longitud', fontsize=12) 
axes[1, 2].set_ylabel('Latitud', fontsize=12) 
axes[1, 2].set_xlim([-87.95,-87.5]) # centro x del grafico
axes[1, 2].set_ylim([41.62,42.05]) # centro y del grafico
plt.show() 

# datos.columns
# Mapas sobre los colegios
fig, axes = plt.subplots(2, 3, figsize=(15, 15)) 
fig.suptitle('Datos Colegios', fontsize=20)
sns.scatterplot(ax=axes[0, 0], data=chi2012, x=chi2012['Longitude'], y=chi2012['Latitude'], hue=chi2012['General Services Route ']) 
axes[0, 0].set_xlabel('Longitud', fontsize=12) 
axes[0, 0].set_ylabel('Latitud', fontsize=12) 
axes[0, 0].set_xlim([-87.95,-87.5]) # centro x del grafico
axes[0, 0].set_ylim([41.62,42.05]) # centro y del grafico
sns.scatterplot(ax=axes[0, 1], data=chi2012, x=chi2012['Longitude'], y=chi2012['Latitude'], hue=chi2012['College Enrollment (number of students) ']) 
axes[0, 1].set_xlabel('Longitud', fontsize=12) 
axes[0, 1].set_ylabel('Latitud', fontsize=12) 
axes[0, 1].set_xlim([-87.95,-87.5]) # centro x del grafico
axes[0, 1].set_ylim([41.62,42.05]) # centro y del grafico
sns.scatterplot(ax=axes[0, 2], data=chi2012, x=chi2012['Longitude'], y=chi2012['Latitude'], hue=chi2012['ISAT Value Add Read']) 
axes[0, 2].set_xlabel('Longitud', fontsize=12) 
axes[0, 2].set_ylabel('Latitud', fontsize=12) 
axes[0, 2].set_xlim([-87.95,-87.5]) # centro x del grafico
axes[0, 2].set_ylim([41.62,42.05]) # centro y del grafico
sns.scatterplot(ax=axes[1, 0], data=chi2012, x=chi2012['Longitude'], y=chi2012['Latitude'], hue=chi2012['ISAT Value Add Math']) 
axes[1, 0].set_xlabel('Longitud', fontsize=12) 
axes[1, 0].set_ylabel('Latitud', fontsize=12) 
axes[1, 0].set_xlim([-87.95,-87.5]) # centro x del grafico
axes[1, 0].set_ylim([41.62,42.05]) # centro y del grafico
sns.scatterplot(ax=axes[1, 1], data=chi2012, x=chi2012['Longitude'], y=chi2012['Latitude'], hue=chi2012['ISAT Exceeding Reading % ']) 
axes[1, 1].set_xlabel('Longitud', fontsize=12) 
axes[1, 1].set_ylabel('Latitud', fontsize=12) 
axes[1, 1].set_xlim([-87.95,-87.5]) # centro x del grafico
axes[1, 1].set_ylim([41.62,42.05]) # centro y del grafico
sns.scatterplot(ax=axes[1, 2], data=chi2012, x=chi2012['Longitude'], y=chi2012['Latitude'], hue=chi2012['ISAT Exceeding Math %']) 
axes[1, 2].set_xlabel('Longitud', fontsize=12) 
axes[1, 2].set_ylabel('Latitud', fontsize=12) 
axes[1, 2].set_xlim([-87.95,-87.5]) # centro x del grafico
axes[1, 2].set_ylim([41.62,42.05]) # centro y del grafico
plt.show() 
# importar mapa geogrfico de chicago
street_map = gpd.read_file("D:/MASTER/Chicago/geo_export_d4f75722-908e-474f-8047-00b0f514eefb.shp")
# tipo de coordenadas
crs = {"init":"espc:4326"}
# se unen longitud y latitud
geometry = [Point(xy) for xy in zip(chi2012["Longitude"], chi2012["Latitude"])]
# crea un dataframa tipo GeoPandas
geo_df = gpd.GeoDataFrame(chi2012, crs = crs, geometry = geometry)
# Se crea el grafico
fig, ax = plt.subplots(figsize=(15,10))
street_map.plot(ax=ax, alpha=0.4,color="grey")
geo_df.plot(column="HARDSHIP INDEX",ax=ax,alpha=0.5, legend=True,markersize=1)
plt.title("Indice de Dificultad", fontsize=15,fontweight="bold")
# latitud y longitud para el mapa
plt.xlim(-87.95,-87.5)
plt.ylim( 41.62,42.05)
plt.show()

#Posiciones geograficas donde se detiene y donde no se detiene
chi2012['Arrest'].replace((True,False),(1,0),inplace = True)
fig, ax = plt.subplots(figsize=(15,15))
street_map.plot(ax=ax, alpha=0.2, color='black')
geo_df[chi2012['Arrest'] ==1].plot(ax=ax, 
                                        markersize=0.2, 
                                        color='blue', 
                                        marker='o', 
                                        label='No detenciones')
geo_df[chi2012['Arrest'] ==0].plot(ax=ax, 
                                        markersize=0.2, 
                                        color='red', 
                                        marker='^', 
                                        label='Detenciones')
plt.legend(prop={'size':10})

#Graficos segun el tipo delictivo primario
datos_thef = chi2012['Primary Type'] == "THEFT"
fig, ax = plt.subplots(figsize=(5,5))
street_map.plot(ax=ax, alpha=0.2, color='black')
geo_df[datos_thef].plot(ax=ax, markersize=0.1,
                                        color='blue', 
                                        marker='o', 
                                        label='Robos')
plt.legend(prop={'size':10})

datos_battery = chi2012['Primary Type'] == "NARCOTICS"
fig, ax = plt.subplots(figsize=(5,5))
street_map.plot(ax=ax, alpha=0.2, color='black')
geo_df[datos_battery].plot(ax=ax, markersize=0.1,
                                        color='red', 
                                        marker='o', 
                                        label='Trafico de Drogas')
plt.legend(prop={'size':10})

datos_thef = chi2012[chi2012['Primary Type'] == "THEFT"]
datos_battery = chi2012[chi2012['Primary Type'] == "NARCOTICS"]

#Mapas de calor
datos_thef.plot.hexbin(x='Longitude', y='Latitude', gridsize=(30,30))
plt.title("Theft")
datos_battery.plot.hexbin(x='Longitude', y='Latitude',gridsize=(30,30))
plt.title("Battery")

#Mapas de calor usando Folium y Jupyter Notebook
import folium
from folium import plugins
from folium.plugins import MarkerCluster,HeatMap, HeatMapWithTime
#Totales
mapa = folium.Map(location=[41.739685,-87.554420],
                    zoom_start = 10) #Coordenadas de Chicago
datos2=datos[datos["Primary Type"]=="THEFT"] 
datos2 = datos[['Latitude', 'Longitude']] 
visualizar = [[row['Latitude'],row['Longitude']] for index, row in datos2.iterrows()]
HeatMap(visualizar, radius=12).add_to(mapa) 
mapa
mapa = folium.Map(location=[41.739685,-87.554420],
                    zoom_start = 10) #Coordenadas de Chicago
datos2=datos[datos["Primary Type"]=="ROBBERY"] 
datos2 = datos[['Latitude', 'Longitude']] 
visualizar = [[row['Latitude'],row['Longitude']] for index, row in datos2.iterrows()]
HeatMap(visualizar, radius=12).add_to(mapa) 
mapa
# Mapas solo eligiendo el año 2012
mapa = folium.Map(location=[41.739685,-87.554420],
                    zoom_start = 10) #Coordenadas de Chicago
chi2012_aux=chi2012[chi2012["Primary Type"]=="THEFT"] 
chi2012_aux = chi2012[['Latitude', 'Longitude']] 
visualizar = [[row['Latitude'],row['Longitude']] for index, row in chi2012_aux.iterrows()]
HeatMap(visualizar, radius=12).add_to(mapa) 
mapa
mapa = folium.Map(location=[41.739685,-87.554420],
                    zoom_start = 10) #Coordenadas de Chicago
chi2012_aux=chi2012[chi2012["Primary Type"]=="ROBBERY"] 
chi2012_aux = chi2012[['Latitude', 'Longitude']] 
visualizar = [[row['Latitude'],row['Longitude']] for index, row in chi2012_aux.iterrows()]
HeatMap(visualizar, radius=12).add_to(mapa) 
mapa
mapa = folium.Map(location=[41.739685,-87.554420],
                    zoom_start = 10) #Coordenadas de Chicago
chi2012_aux=chi2012[chi2012["Primary Type"]=="NARCOTICS"] 
chi2012_aux = chi2012[['Latitude', 'Longitude']] 
visualizar = [[row['Latitude'],row['Longitude']] for index, row in chi2012_aux.iterrows()]
HeatMap(visualizar, radius=12).add_to(mapa) 
mapa
#Detenciones
mapa = folium.Map(location=[41.739685,-87.554420],
                    zoom_start = 10) #Coordenadas de Chicago
chi2012_aux=chi2012[chi2012["Arrest"]==1] 
chi2012_aux = chi2012[['Latitude', 'Longitude']] 
visualizar = [[row['Latitude'],row['Longitude']] for index, row in chi2012_aux.iterrows()]
HeatMap(visualizar, radius=12).add_to(mapa) 
mapa

mapa = folium.Map(location=[41.739685,-87.554420],
                    zoom_start = 10) #Coordenadas de Chicago
chi2012_aux=chi2012[chi2012["Arrest"]==0] 
chi2012_aux = chi2012[['Latitude', 'Longitude']] 
visualizar = [[row['Latitude'],row['Longitude']] for index, row in chi2012_aux.iterrows()]
HeatMap(visualizar, radius=12).add_to(mapa) 
mapa
# Delito Grave
mapa = folium.Map(location=[41.739685,-87.554420],
                    zoom_start = 10) #Coordenadas de Chicago
chi2012_aux=chi2012[chi2012["Delito Grave"]==0] 
chi2012_aux = chi2012[['Latitude', 'Longitude']] 
visualizar = [[row['Latitude'],row['Longitude']] for index, row in chi2012_aux.iterrows()]
HeatMap(visualizar, radius=12).add_to(mapa) 
mapa

mapa = folium.Map(location=[41.739685,-87.554420],
                    zoom_start = 10) #Coordenadas de Chicago
chi2012_aux=chi2012[chi2012["Delito Grave"]==1] 
chi2012_aux = chi2012[['Latitude', 'Longitude']] 
visualizar = [[row['Latitude'],row['Longitude']] for index, row in chi2012_aux.iterrows()]
HeatMap(visualizar, radius=12).add_to(mapa) 
mapa

# Visualizar el numero de delitos primarios, los ultimos delitos tienen pocos valores,
# se decide agrupar los 10 ultimos en una nueva variable (OTROS)
plt.figure(figsize=(15,10))
plt.title('Numero de delitos por tipo')
plt.ylabel('Tipo de Delito')
plt.xlabel('Numero de delitos')
datos.groupby([datos['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh', color=["royalblue","blue"])
plt.show()
# Se crea una nueva columna vacia donde se guardara la suma
datos["Suma"] = datos.apply(lambda _: '0', axis=1)
tipos_delitos=datos.groupby(['Primary Type'])["Suma"].size().reset_index()
tipos_delitos=tipos_delitos.sort_values(["Suma"], ascending=[False])
ultimos=tipos_delitos.tail(10)
# Los ultimos 10 se guardan en el campo Otros, por ultimo se borra la columna Suma
datos.loc[datos['Primary Type'].isin(ultimos['Primary Type']), 'Primary Type'] = 'OTROS'
datos=datos.drop(["Suma"],axis=1)
# # Se eliminan variables
# del tipos_delitos, ultimos
# Se graba las modificaciones realizadas cuando se agruparon los delitos (OTROS)
datos.to_csv("D:/MASTER/Chicago/datos2.csv")

# Visualizar tipos de delitos
plt.figure(figsize=(15,10))
plt.title('Suma de delitos')
plt.ylabel('Tipo de Delitos')
plt.xlabel('Numero de Delitos')
datos.groupby([datos['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh', color=["royalblue","blue"])
plt.show()

# Numero de delitos por distrito
grafico = datos.groupby(['Primary Type',"District"]).size().reset_index().pivot(columns="District", 
                        index='Primary Type', values=0)
grafico.plot.bar(figsize=(20,10),stacked=True, title="Delitos por Distrito",fontsize=10)
# x=datos["Primary Type"].unique()
# x.sort()
# print(x)
#Graficas Delitos por franja del dia
leyenda = ['ARSON', 'ASSAULT', 'BATTERY', 'BURGLARY' ,'CRIM SEXUAL ASSAULT',
 'CRIMINAL DAMAGE', 'CRIMINAL TRESPASS', 'DECEPTIVE PRACTICE', 'GAMBLING',
 'INTERFERENCE WITH PUBLIC OFFICER', 'LIQUOR LAW VIOLATION',
 'MOTOR VEHICLE THEFT', 'NARCOTICS', 'OFFENSE INVOLVING CHILDREN',
 'OTHER OFFENSE', 'OTROS', 'PROSTITUTION', 'PUBLIC PEACE VIOLATION' ,'ROBBERY',
 'SEX OFFENSE', 'THEFT', 'WEAPONS VIOLATION']
grafico = datos.groupby(['Primary Type',"Franja Dia"]).size().reset_index().pivot(columns="Franja Dia", 
                        index='Primary Type', values=0)
grafico.plot.line(figsize=(20,10),title="Delitos por Parte del Dia")

#Delitos fin de semana

df_plot = datos.groupby(['Primary Type','Fin Semana']).size().reset_index().pivot(columns='Fin Semana', 
                        index='Primary Type', values=0)
ax =df_plot.plot(kind='bar',title='Delitos por Fin de Semana')
ax.set(ylabel='Numero de delitos', xlabel='Tipo de delitos') 

#Media de delitos si es fin de semana o no (fin de semana son 3 dias)
media_fin_semana = df_plot.groupby(['Primary Type']).mean().reset_index()
media_fin_semana["Si"]=media_fin_semana["Si"].apply(lambda x: x/3)
media_fin_semana["No"]=media_fin_semana["No"].apply(lambda x: x/4)

fig, ax = plt.subplots()
plt.ylabel('%')
plt.title('Variacion delitos si es fin de semana o no (Fin Semana= Viernes, Sabado y Domingo')
plt.xlabel('Tipos Delitos')
ax.set_xticklabels(leyenda)
plt.xticks(rotation=90)
plt.xticks(np.arange(0, 22, 1.0))
fig=plt.gcf()
fig.set_size_inches(15,5)
ax.plot(media_fin_semana['Si'].values, color = 'blue', label = 'Si Fin Semana')
ax.plot(media_fin_semana['No'].values, color = 'brown', label = 'No Fin Semana')
plt.legend()

media_fin_semana.plot(x="Primary Type",y=["Si","No"],kind="barh",width=0.7,figsize=(15,10),title="Media de delitos dependiendo si es fin de semana o no")

fig, ax = plt.subplots()
fig.set_size_inches(15,12)
g=media_fin_semana["Primary Type"]
v1=media_fin_semana["Si"]
v2=media_fin_semana["No"]
ax.set_title('Delitos dependiendo si es fin de semana o no')
ax.bar(g,v1,label="Si Fin de Semana")
ax.bar(g,v2,bottom =v1, label="No fin de Semana")
for bar in ax.patches:
  ax.text(bar.get_x() + bar.get_width() / 2,
          bar.get_height() / 2 + bar.get_y(),
          round(bar.get_height()), ha = 'center',
          color = 'black', weight = 'bold', size = 7)
ax.legend()
plt.xticks(rotation=90)

#Delitos por dia
delito_dia=datos.groupby(['Primary Type',"Dia Semana"]).size().reset_index().pivot(columns="Dia Semana", 
                        index='Primary Type', values=0)
delito_dia.plot.bar(figsize=(18,6),title="Delitos por Dia de la Semana (lunes=0, domingo=6)")


#Delitos segun partidos
si_partido=datos.query("Partido==1")
total_si_partido=len(si_partido)
no_partido=datos.query("Partido==0")
total_no_partido=len(no_partido)

si_partido=si_partido.groupby(['Primary Type',"Partido"]).size().reset_index().pivot(columns="Partido", 
                        index='Primary Type', values=0)
si_partido=pd.DataFrame(si_partido)
si_partido.rename(columns={1:"Total"}, inplace=True)
si_partido["Media_Delitos_si_partido"]=si_partido["Total"]/(total_si_partido)

no_partido=no_partido.groupby(['Primary Type',"Partido"]).size().reset_index().pivot(columns="Partido", 
                        index='Primary Type', values=0)
no_partido=pd.DataFrame(no_partido)
no_partido.rename(columns={0:"Total"}, inplace=True)
no_partido["Media_Delitos_no_partido"]=no_partido["Total"]/(total_no_partido)

partidos_conjunto=datos.groupby(['Primary Type',"Partido"]).size().reset_index().pivot(columns="Partido", 
                        index='Primary Type', values=0)

partidos_conjunto.plot.bar(figsize=(18,6),title="Delitos segun Partidos (0=No, 1=Si)")

datos_partidos=pd.merge(si_partido, no_partido, on='Primary Type', how='left')
 
fig, ax = plt.subplots()
plt.ylabel('%')
plt.title('Variacion delitos si hay partido o no')
plt.xlabel('Tipos Delitos')
ax.set_xticklabels(leyenda)
plt.xticks(rotation=90)
plt.xticks(np.arange(0, 22, 1.0))
fig=plt.gcf()
fig.set_size_inches(15,5)
ax.plot(datos_partidos['Media_Delitos_si_partido'].values, color = 'Blue', label = 'Si Partido')
ax.plot(datos_partidos['Media_Delitos_no_partido'].values, color = 'red', label = 'No Partido')
plt.legend()

#Festivos
partidos_conjunto=datos.groupby(['Primary Type',"Festivo"]).size().reset_index().pivot(columns="Festivo", 
                        index='Primary Type', values=0)
partidos_conjunto.plot.bar(figsize=(18,6),title="Delitos segun Festivos (0=No, 1=Si)",color=["green","orangered"])

si_festivo=datos.query("Festivo==1")
total_si_festivo=len(si_festivo)
no_festivo=datos.query("Festivo==0")
total_no_festivo=len(no_festivo)

si_festivo=si_festivo.groupby(['Primary Type',"Festivo"]).size().reset_index().pivot(columns="Festivo", 
                        index='Primary Type', values=0)
si_festivo=pd.DataFrame(si_festivo)
si_festivo.rename(columns={1:"Total"}, inplace=True)
si_festivo["Media_Delitos_si_festivo"]=si_festivo["Total"]/(total_si_festivo)

no_festivo=no_festivo.groupby(['Primary Type',"Festivo"]).size().reset_index().pivot(columns="Festivo", 
                        index='Primary Type', values=0)
no_festivo=pd.DataFrame(no_festivo)
no_festivo.rename(columns={0:"Total"}, inplace=True)
no_festivo["Media_Delitos_no_festivo"]=no_festivo["Total"]/(total_no_festivo)

datos_festivo=pd.merge(si_festivo, no_festivo, on='Primary Type', how='left')

fig, ax = plt.subplots()
plt.ylabel('%')
plt.title('Variacion delitos si es festivo o no')
plt.xlabel('Tipos Delitos')
ax.set_xticklabels(leyenda)
plt.xticks(rotation=90)
plt.xticks(np.arange(0, 22, 1.0))
fig.set_size_inches(20,5)
ax.plot(datos_festivo['Media_Delitos_si_festivo'].values, color = 'Green', label = 'Si Festivo')
ax.plot(datos_festivo['Media_Delitos_no_festivo'].values, color = 'Orange', label = 'No Festivo')
plt.legend()

#Delitos segun temperatura

temperatura = datos.groupby(["Temperatura(C)"],as_index=False).agg({"Primary Type":"count"})
temperatura=temperatura.rename(columns={"Primary Type":'Delitos_Totales'})
sns.regplot(x = "Temperatura(C)", y = "Delitos_Totales", data = temperatura, scatter_kws = {"alpha" : 0.1});
temperatura=temperatura.rename(columns={"Temperatura(C)":"Temperatura"})
from patsy import dmatrices
import statsmodels.api as sm
y, X = dmatrices('Temperatura~Delitos_Totales', data=temperatura, return_type='dataframe')
mod = sm.OLS(y, X)
res = mod.fit()
print(res.summary())
fig = sm.graphics.plot_regress_exog(res, "Delitos_Totales")
fig.tight_layout(pad=1.0)
# datos=datos.drop(["Latitude","Longitude"],axis=1)
#CORRELACIONES (Pearson)
# Se factoriza Primary Type para pasarle a numerico
# y mostralo con corr
datos['Primary Type'] = pd.factorize(datos["Primary Type"])[0] 
plt.figure(figsize=(30,30))
cor = datos.corr().round(2)
mask = np.triu(np.ones_like(cor, dtype=bool))
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds,vmax=1, vmin=-1,mask=mask,center=0)

#Correlation con la variable objetivo
corr= abs(cor['Delito Grave'])
#Se selecciona las variables con mayores correlaciones absolutas
variables_relevantes = corr[corr>0.06]
variables_relevantes
