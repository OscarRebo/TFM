# Importar modulos
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import seaborn as sns
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point
import re
from sklearn.preprocessing import LabelEncoder
#**************BASE DE DATOS SOBRE DELITOS EN CHICAGO***************************
#Datos sobre el crimen desde el 2008 al 2012
crimen = pd.read_csv('D:/MASTER/Chicago/chicago2008_2012.csv')
print(crimen.head(10))
#Se elimina la primera columna que no tiene nombre y no aporta valor
crimen=crimen.drop(crimen.columns[0], axis=1)
print ("Tipos de datos")
print ("**************")
crimen.info()
# Variables continuas y categoricas
continuas=crimen._get_numeric_data().columns
print("Las variables continuas son :",list(continuas))
print("Las variables categoricas son :",list(set(crimen.columns)-set(continuas)))

#*****Analisis de Datos******
#1- Se analizan y eliminan columnas irrelevantes o repetidas
crimen[['IUCR']].groupby('IUCR').size()
crimen[['FBI Code']].groupby('FBI Code').size()
crimen[['Description']].groupby('Description').size()
crimen[['Location Description']].groupby('Location Description').size()
lugares=crimen.groupby([crimen['Location Description']]).size().sort_values(ascending=False)
print(lugares.head(160))
crimen[['Arrest']].groupby('Arrest').size()
crimen[['Domestic']].groupby('Domestic').size()
crimen[['Beat']].groupby('Beat').size()
crimen[['District']].groupby('District').size()
crimen[['Ward']].groupby('Ward').size()
crimen[['Community Area']].groupby('Community Area').size()
crimen[['FBI Code']].groupby('FBI Code').size()
crimen[['Year']].groupby('Year').size()
crimen[['Latitude']].groupby('Latitude').size()
crimen[['Longitude']].groupby('Longitude').size()
crimen[['Community Areas']].groupby('Community Areas').size()
crimen[['Census Tracts']].groupby('Census Tracts').size()
crimen[['Wards']].groupby('Wards').size()
crimen[['Boundaries - ZIP Codes']].groupby('Boundaries - ZIP Codes').size()
crimen[['Police Districts']].groupby('Police Districts').size()
crimen[['Police Beats']].groupby('Police Beats').size()
crimen=crimen.drop(["ID","Case Number","Updated On", "Historical Wards 2003-2015","Description",
                    "Zip Codes","Location","X Coordinate","Y Coordinate","IUCR","FBI Code",
                    "Community Areas",'Census Tracts','Wards','Boundaries - ZIP Codes'],axis=1)
# #2 Tratamiento de datos
# Se separa campo fecha, se pasa a 24 horas y se eliminan los segundos
# Se comprueba si tiene AM o PM en la fecha (hay fechas ya en formato 24 horas)
#Se crea el dia de la semana (lunes=0, domingo=6)
fechat=crimen["Date"]
print((fechat))
hora=[]
fecha=[]
dias_semana=[]
mes=[]
solohora=[]
solodia=[]
for i in range(len(fechat)):  
  fecha1=fechat[i]
  fechacon=fecha1[-2:]
  # print(fechacon)
  if (fechacon=="AM" or fechacon=="PM"):
    f, h,am_pm = fecha1.split(" ")
    h=h[:5] 
    horatotal=h+" "+am_pm
    hora_am_pm = datetime.strptime(horatotal, "%I:%M %p")
    hora24 = datetime.strftime(hora_am_pm, "%H:%M")
  else:
    f, h= fecha1.split(" ")  
    hora24=h
  h1=datetime.strftime(hora_am_pm, "%H:%M")[:2]
  solohora.append(h1)
  d1=f[3:5]
  solodia.append(d1)  
  dia_semana=datetime.strptime(f,'%m/%d/%Y')
  dia_semana2=dia_semana.weekday() 
  dias_semana.append(dia_semana2) 
  hora.append(hora24)
  f2=datetime.strptime(f,"%m/%d/%Y").strftime('%d/%m/%Y')
  fecha.append(f2)
  mes1=dia_semana.strftime("%m")
  mes.append(mes1)
crimen["Hora completa"]=hora
crimen["Hora"]=solohora
crimen["Fecha"]=fecha
crimen["Dia Semana"]=dias_semana
crimen["Mes"]=mes
crimen["Dia"]=solodia
print(crimen["Hora"])
#Se elimina el campo Date y variables usadas
# crimen=crimen.drop(["Date"],axis=1)
del am_pm, dia_semana, dia_semana2,f,h, hora, hora24, hora_am_pm, horatotal, fechacon,fecha1,fecha,dias_semana
#Se crea variable fin_semana (dias 4,5 y 6 correspondientes a viernes, sabado y domingo) y se elimina el dia_semana
# Se define como 0 si es fin de semana y como 1 si no lo es
crimen["Fin Semana"]=crimen["Dia Semana"].apply(lambda x: 1 if(x==4 or x==5 or x==6)else 0)
# Hay muchas horas diferentes, se agrupan por partes del dia 
# Se añade columna Franja Dia, correspondiente a las diferentes partes del dia
# 0=madrugada, 1=Mañana, 2=Tarde y 3= Noche
Franja_Dia=[]
p=[]
for i in range(len(fechat)):
    t1=crimen["Hora completa"][i] 
    t1=datetime.strptime(t1,"%H:%M")
    hora=int(t1.hour)
    if (hora>0 and hora<6):
        p="Madrugada"
    elif (hora>= 6 and hora<12):
        p="Mañana"
    elif (hora>= 12 and hora<18):
        p="Tarde"
    else:
        p="Noche"     
    Franja_Dia.append(p)
crimen["Franja Dia"]=Franja_Dia 

#Agrupar segun la gravedad del delito
delitos_graves=["ARSON", "ASSAULT", "BATTERY", "CRIM SEXUAL ASSAULT","CRIMINAL DAMAGE", 
                    "CRIMINAL TRESPASS", "HOMICIDE", "ROBBERY"]
crimen['Delito Grave'] = np.where(crimen["Primary Type"].isin(delitos_graves), 1, 0) 

# Agrupar lugares
def lugares(item):
    lugares_primeros=['STREET','RESIDENCE','SIDEWALK','APARTMENT']
    if item not in lugares_primeros:
        return "OTROS"
    else:
        return item
crimen["Lugar"]=crimen["Location Description"].apply(lugares)

#Agrupar longitud y latitud
paso = 0.01
to_bin = lambda x: np.floor(x / paso) * paso
crimen["latPaso"] = to_bin(crimen["Latitude"])
crimen["lonPaso"] = to_bin(crimen["Longitude"])

# Extraer valor al campo Block
Block_temp = []
Block_num = []
Block_calle = []
temp_df = crimen[['Block']]
for index, row in temp_df.iterrows():
    Block_temp.append(row['Block'])

patron1 = "X (.+)"
patron3 = r'\d+'

i=0
temp=[]
for data in Block_num:
    if data == 0:
        temp.append(i)
    i+=1
#Se seleccionan los campos que no sean XX UNKNOWN
Block_temp1 = [s for s in Block_temp if s != 'XX  UNKNOWN']
i=0
for data in Block_temp1:  
    # print(data)
    Block_calle.append(re.findall(patron1, data)[0])
    Block_num.append(int(re.findall(patron3, data)[0]))
    i+=1
# Eliminar campos on el valor desconocido
indexNames = crimen[ crimen['Block'] == 'XX  UNKNOWN'].index
crimen.drop(indexNames , inplace=True)
# Añade nuevos campos
crimen["Block_calle"] = Block_calle
crimen["Block_Num"] = Block_num
# Codifican a valores numericos
le = LabelEncoder()
Block_array = le.fit_transform(crimen['Block'])
Block_calle_1 = le.fit_transform(crimen['Block_calle'])
crimen = crimen.drop(columns=['Block', 'Block_calle'])
crimen['Block'] = Block_array
crimen['Block_calle'] = Block_calle_1
#3- Variables sin valores
crimen.isnull().sum().reset_index()
# Hay muchas columnas con variables nulas, las eliminamos
crimen = crimen.dropna()
crimen['Block_Num'] = crimen.Block_Num.astype(int)
crimen['Beat'] = crimen.Beat.astype(int)
crimen['District'] = crimen.District.astype(int)
# crimen["Franja Dia"].unique() 
# Se elimina, hora y variables usadas
del p, t1, Franja_Dia, hora , i, fechat
# 4- Valores outliders o atipicos
# Diagrama de Caja
sns.boxplot(x=crimen["Longitude"])

sns.boxplot(x=crimen["Latitude"])
# Diagrama de Dispersion
fig, ax = plt.subplots(figsize=(6,12)) 
ax.scatter(crimen["Longitude"], crimen["Latitude"]) 
ax.set_xlabel('Longitud') 
ax.set_ylabel('Latitud') 
plt.show()

# # Puntuacion Zeta
# # from scipy import stats
# # z = np.abs(stats.zscore(crimen)) 
# # print(z)
# # print (np.where(z>3))

# #IQR 
Q1 = crimen.quantile(0.25)
Q3 = crimen.quantile(0.75)
IQR = Q3 - Q1
# print(IQR)
superior=Q3+1.5*IQR
inferior=Q1-1.5*IQR
# print ("*******Limite superior IQR********\n",superior)
# print ("*******Limite inferior IQR********\n",inferior)

crimen_sin_outliers = crimen[~((crimen < (inferior)) | (crimen > (superior))).any(axis=1)]

crimen=crimen_sin_outliers
crimen=crimen.drop(["Domestic"],axis=1)
# del crimen_sin_outliers
crimen.info()
crimen.columns
crimen.to_csv("D:/MASTER/Chicago/crimen.csv")