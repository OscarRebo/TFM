# Importar modulos
import pandas as pd 
#####UNION DATASETS ####################
#Se unen las bases por el campo comun Fecha
#Se elimina la primera columna que es el indice
crimen = pd.read_csv('D:/MASTER/Chicago/crimen2.csv')
crimen=crimen.drop(crimen.columns[0], axis=1)
socieconomia = pd.read_csv("D:/MASTER/Chicago/socieconomia.csv")
socieconomia=socieconomia.drop(socieconomia.columns[0], axis=1)
colegios = pd.read_csv('D:/MASTER/Chicago/colegios.csv')
nba=pd.read_csv("D:/MASTER/Chicago/Datos/nba.csv")
nba=nba.drop(nba.columns[0], axis=1)
nhl=pd.read_csv("D:/MASTER/Chicago/Datos/nhl.csv")
nhl=nhl.drop(nhl.columns[0], axis=1)
nfl=pd.read_csv("D:/MASTER/Chicago/Datos/nfl.csv")
nfl=nfl.drop(nfl.columns[0], axis=1)
festivos=pd.read_csv("D:/MASTER/Chicago/Datos/Festivos.csv")
festivos=festivos.drop(festivos.columns[0], axis=1)
tiempo=pd.read_csv("D:/MASTER/Chicago/Datos/tiempo.csv")
datos = pd.merge(crimen, socieconomia, on='Community Area', how='inner')
datos= pd.merge(datos, colegios, on='Community Area', how='inner')
# datos=pd.merge(datos, nba, on='Fecha', how='left')
# datos=pd.merge(datos, nhl, on='Fecha', how='left')
# datos=pd.merge(datos, nfl, on='Fecha', how='left')
datos=pd.merge(datos, tiempo, on='Fecha', how='left')
datos=pd.merge(datos, festivos, on='Fecha', how='left')
#Se rellenan los dias que no hay partidos con cero
# datos["PartidoNBA"]=datos["PartidoNBA"].apply(lambda x: 1 if (x==1)else 0)
# datos["PartidoNHL"]=datos["PartidoNHL"].apply(lambda x: 1 if (x==1)else 0)
# datos["PartidoNFL"]=datos["PartidoNFL"].apply(lambda x: 1 if (x==1)else 0)

datos["Festivo"]=datos["Festivo"].apply(lambda x: 1 if (x==1)else 0)
partidos=pd.merge(nba, nhl, on='Fecha', how='outer')
partidos=pd.merge(partidos, nfl, on='Fecha', how='outer')
longitud=partidos["Fecha"]
partido=[]
for i in range(len(longitud)):
  campo=partidos.iloc[i]
  if ((campo["PartidoNBA"]==1) or (campo["PartidoNFL"]==1) or (campo["PartidoNHL"]==1)):
    p=1
  else:
    p=0
  partido.append(p) 
partidos["Partido"]=partido
partidos=partidos.drop(["PartidoNBA","PartidoNHL","PartidoNFL"],axis=1)
# partidos.fillna(0,inplace=True)
datos=pd.merge(datos, partidos, on='Fecha', how='left')
datos["Partido"]=datos["Partido"].apply(lambda x: 1 if (x==1)else 0)
del crimen, socieconomia, colegios, nba,nhl,nfl,tiempo, festivos, partidos
datos.columns
datos.info()
print(datos.head())
#Valores faltantes
print(datos.isnull().sum())
datos.to_csv("D:/MASTER/Chicago/datos2.csv")
