# Importar modulos
import pandas as pd 
from datetime import datetime
from time import gmtime, strftime
#**************BASE DE DATOS DATOS PARITDO NHL**************************
# https://www.kaggle.com/datasets/martinellis/nhl-game-data
nhl=pd.read_csv("D:/MASTER/Chicago/nhl/game.csv")
nhl.columns
nhl=nhl.query("home_team_id==16")
nhl1=[]
nhl1=nhl[(nhl['season'] > 20082009) & (nhl['season'] < 20112012)]
# nhl1=nhl1[(nhl1['date_time_GMT'] > '2008-01-01') & (nhl1['date_time_GMT'] < '2012-12-31')]
nhl1.columns
nhl1["PartidoNHL"]=nhl1.apply(lambda x:1)
nhl1["PartidoNHL"]=nhl1["PartidoNHL"].apply(lambda x:1)
nhl1["Fecha"]=nhl1["date_time_GMT"]
nhl1=nhl1.reset_index(drop=True)
fechat=nhl1["Fecha"]
fecha=[]
mes=[]
for i in range(len(fechat)):  
    fecha1=fechat[i]
    f, h = fecha1.split("T")
    print(i)
    print(fecha1)
    fecha2 = datetime.strptime(f,"%Y-%m-%d").strftime('%d/%m/%Y')
    fecha.append(fecha2)
nhl1["Fecha"]=fecha
nhl1=nhl1.drop(['game_id', 'season', 'type', 'date_time_GMT', 'away_team_id',
       'home_team_id', 'away_goals', 'home_goals', 'outcome',
       'home_rink_side_start', 'venue', 'venue_link', 'venue_time_zone_id',
       'venue_time_zone_offset', 'venue_time_zone_tz'],axis=1)
# print ("Tipos de datos")
# print ("**************")
nhl1.info()
# ANALISIS DE VARIABLES
#1- No existen valores faltantes
nhl1.isnull().sum().reset_index()
nhl1.to_csv("D:/MASTER/Chicago/Datos/nhl.csv")