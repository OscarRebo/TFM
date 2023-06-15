# Importar modulos
import pandas as pd 
#**************BASE DE DATOS DATOS SOCIOECONOMICOS EN CHICAGO***************************
socioeconomia1=pd.read_csv('D:/MASTER/Chicago/Census_Data_-_Selected_socioeconomic_indicators_in_Chicago__2008___2012.csv')
#Se renombra el campo Community area de la base socioeconomico1 para que coincida en todas las bases
socioeconomia = socioeconomia1.rename(columns={'Community Area Number': 'Community Area'})
#Se elimina la ultima fila
socioeconomia = socioeconomia.iloc[:-1]
print ("Tipos de datos")
print ("**************")
socioeconomia.info()
# ANALISIS DE VARIABLES
#1- No existen valores faltantes
socioeconomia.isnull().sum().reset_index()
#Se eliminan campos que no aportan valor al modelo
socioeconomia=socioeconomia.drop(["COMMUNITY AREA NAME"],axis=1)
socioeconomia.to_csv("D:/MASTER/Chicago/socieconomia.csv")