# Importar modulosç
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
# !pip install pmdarima
from pmdarima.arima import ADFTest , auto_arima
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6
import warnings
warnings.filterwarnings('ignore')
#Cargar datos
df=pd.read_csv("D:/MASTER/Chicago/datos2.csv")
# datos=datos.drop(datos.columns[0], axis=1)
df.columns
df.index = pd.DatetimeIndex(df['Fecha'])
df.sort_index(inplace=True)
df.info()
df = pd.pivot_table(df[["Primary Type", "Arrest"]], index=df[["Fecha", "Primary Type"]], aggfunc="count")
df.reset_index(inplace=True)
df.rename(columns={"Arrest":"Delitos Totales"}, inplace=True)
df.index = pd.DatetimeIndex(df['Fecha'])
df.sort_index(inplace=True)
df=df.drop(["Primary Type","Fecha"],axis=1)
df.info()
df.head(10)
df_dia=[]
df_dia=df
df_dia.head()
fig_size = (20,5)
df_dia.plot(figsize=fig_size, title="Media diaria ")
plt.show()
df_dia["Delitos Totales"].sort_values(ascending=False).head()
df_mes = pd.DataFrame(df["Delitos Totales"].resample('M').size())
df_mes.plot(
    figsize=fig_size, 
    title="Media Mensual " )
plt.show()

adf=ADFTest(alpha=0.05)
adf.should_diff(df_mes["Delitos Totales"])

start=int(df_mes.shape[0]*0.55)
train=df_mes[:start]
test=df_mes[start:]
#P,D Y Q (2,1,0)
modelo=auto_arima(train["Delitos Totales"],start_p=0,d=1,start_q=0,
          max_p=10,max_d=10,max_q=10, start_P=2,
          D=1, start_Q=0, max_P=10,max_D=10,
          max_Q=10, m=12, seasonal=True,
          error_action='warn',trace=True,
          supress_warnings=True,stepwise=True,
          random_state=40,n_fits=100)
prediccion = pd.DataFrame(modelo.predict(n_periods = train.shape[0]),index=test.index)
prediccion.columns = ['prediccion']
plt.figure(figsize=fig_size)
plt.plot(df_mes,label="Entrenamiento")
plt.plot(test,label="Test")
plt.plot(prediccion,label="Prediccion")
plt.legend(loc = 'upper right')
plt.show()

modelo.summary()