# Importar modulos
import pandas as pd 
from datetime import datetime
#**************BASE DE DATOS PARITDOS NBA***************************
#https://www.kaggle.com/datasets/nathanlauga/nba-games?resource=download.
nba=pd.read_csv("D:/MASTER/Chicago/nba/games.csv")
nba=nba.query("HOME_TEAM_ID==1610612741")
nba.columns
nba1=[]
nba1=nba[(nba['GAME_DATE_EST'] > '2008-01-01') & (nba['GAME_DATE_EST'] < '2012-12-31')]
nba1.columns
nba1["PartidoNBA"]=nba1.apply(lambda x:1)
nba1["PartidoNBA"]=nba1["PartidoNBA"].apply(lambda x:1)
nba1["Fecha"]=nba1["GAME_DATE_EST"]
nba1=nba1.reset_index(drop=True)
fechat=nba1["Fecha"]
fecha=[]
mes=[]
for i in range(len(fechat)):  
    fecha1=fechat[i]
    print(i)
    print(fecha1)
    fecha2 = datetime.strptime(fecha1,'%Y-%m-%d').strftime('%d/%m/%Y')
    fecha.append(fecha2)
nba1["Fecha"]=fecha
nba1=nba1.drop(['GAME_DATE_EST', 'GAME_ID', 'GAME_STATUS_TEXT', 'HOME_TEAM_ID',
               'VISITOR_TEAM_ID', 'SEASON', 'TEAM_ID_home', 'PTS_home', 'FG_PCT_home',
                'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'TEAM_ID_away',
                 'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away',
                 'REB_away', 'HOME_TEAM_WINS'],axis=1)
# print ("Tipos de datos")
# print ("**************")
nba1.info()
# ANALISIS DE VARIABLES
#1- No existen valores faltantes
nba1.isnull().sum().reset_index()
nba1.to_csv("D:/MASTER/Chicago/Datos/nba.csv")