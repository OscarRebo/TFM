# Importar modulos
import pandas as pd 
from datetime import datetime
from time import gmtime, strftime
#**************BASE DE DATOS DATOS PARITDO NFL**************************
#https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data?select=spreadspoke_scores.csv.
nfl=pd.read_csv("D:/MASTER/Chicago/nfl/spreadspoke_scores.csv")
nfl.columns
nfl=nfl[nfl["team_home"]=="Chicago Bears"]
nhl1=[]
nfl1=nfl[(nfl['schedule_season'] >= 2008) & (nfl['schedule_season'] <= 2012)]

nfl1.columns
nfl1["PartidoNFL"]=nfl1.apply(lambda x:1)
nfl1["PartidoNFL"]=nfl1["PartidoNFL"].apply(lambda x:1)
nfl1["Fecha"]=nfl1["schedule_date"]
nfl1=nfl1.reset_index(drop=True)
fechat=nfl1["Fecha"]
fecha=[]
mes=[]
for i in range(len(fechat)):  
    fecha1=fechat[i]
    print(i)
    print(fecha1)
    fecha2 = datetime.strptime(fecha1,'%m/%d/%Y').strftime('%d/%m/%Y')
    fecha.append(fecha2)
nfl1["Fecha"]=fecha
nfl1=nfl1.drop(['schedule_date', 'schedule_season', 'schedule_week', 'schedule_playoff',
       'team_home', 'score_home', 'score_away', 'team_away',
       'team_favorite_id', 'spread_favorite', 'over_under_line', 'stadium',
       'stadium_neutral', 'weather_temperature', 'weather_wind_mph',
       'weather_humidity', 'weather_detail'],axis=1)
# print ("Tipos de datos")
# print ("**************")
nfl1.info()
# ANALISIS DE VARIABLES
#1- No existen valores faltantes
nfl1.isnull().sum().reset_index()
nfl1.to_csv("D:/MASTER/Chicago/Datos/nfl.csv")