#https://github.com/harris-ippp/lectures/blob/master/09-weather/weather_crime.ipynb
#https://www.kaggle.com/code/vgalan/crimes-in-chicago-my-first-eda
# Importar modulos
import pandas as pd 
#####UNION DATASETS ####################
#Se unen las bases por el campo comun Fecha
#Se elimina la primera columna que es el indice
crimen = pd.read_csv('D:/MASTER/Chicago/crimen.csv')
crimen=crimen.drop(crimen.columns[0], axis=1)
tiempo=pd.read_csv("D:/MASTER/Chicago/tiempo/archive.csv")
tiempo["Temperatura(C)"]=tiempo["temperature_2m (°C)"]
cw = pd.concat([tiempo, crimen], axis = 1).dropna()
from matplotlib import pyplot as plt 
import seaborn as sns

agrupar = cw.groupby(["Temperatura(C)"],as_index=False).agg({"Primary Type":"count"})
agrupar=agrupar.rename(columns={"Primary Type":'Delitos Totales'})


sns.regplot(x = "Temperatura(C)", y = "Delitos Totales", data = agrupar, scatter_kws = {"alpha" : 0.1});

