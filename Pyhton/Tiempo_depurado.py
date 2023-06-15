# Importar modulos
import pandas as pd 
from datetime import datetime
from time import gmtime, strftime
#**************BASE DE DATOS DATOS TIEMPO**************************
# Datos extraidos de
# https://open-meteo.com/en/docs/historical-weather-api
tiempo=pd.read_csv("D:/MASTER/Chicago/tiempo/archive.csv")
tiempo.columns
fechat=tiempo["time"]
fecha=[]
mes=[]
for i in range(len(fechat)):  
    fecha1=fechat[i]
    f, h = fecha1.split("T")    
    fecha2 = datetime.strptime(f,"%Y-%m-%d").strftime('%d/%m/%Y')
    fecha.append(fecha2)
tiempo["Fecha"]=fecha
tiempo=tiempo.drop(tiempo.columns[0], axis=1)
# print ("Tipos de datos")
# print ("**************")
tiempo.info()
tiempo["Temperatura(C)"]=tiempo["temperature_2m (°C)"]
tiempo=tiempo.drop(["temperature_2m (°C)"],axis=1)
#Calcula la media de temperatura por dia
agrupado = tiempo.groupby(["Fecha"]).mean()
agrupado.head(50)
tiempo2=round(agrupado,2)
tiempo2.columns
tiempo["Temperatura(C)"]=tiempo2
# ANALISIS DE VARIABLES
#1- No existen valores faltantes
tiempo2.isnull().sum().reset_index()
tiempo2.to_csv("D:/MASTER/Chicago/Datos/tiempo.csv")
