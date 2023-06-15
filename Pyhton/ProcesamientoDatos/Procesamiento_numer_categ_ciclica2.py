# Importar modulos
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
datos = pd.read_csv('D:/MASTER/Chicago/datos2.csv')
#Se elimina la primera columna y datos que no aportan
datos=datos.drop(datos.columns[0], axis=1)
datos=datos.drop(["Date","Hora completa","Latitude","Longitude","Location Description"],axis=1)
print(datos.head(10))
##############################
numericas=datos._get_numeric_data().columns
print("Las variables numericas son :",list(numericas))
print("Las variables categoricas son :",list(set(datos.columns)-set(numericas)))
########### VARIABEL OBJETIVO #################
# Primary Type
########## CATEGORICAS ########################
###############################################
#1-Nominales (One Hot Encoding)
datos = pd.get_dummies(datos, columns=['Lugar']) 
#2- Ordinales (OrdinalEncoding)
ord = OrdinalEncoder()
datos['Franja Dia'] = ord.fit_transform(datos['Franja Dia'].values.reshape(-1,1))
# Fin de semana tiene los valores Si y No, lo pasamos a binario
datos["Fin Semana"]=datos["Fin Semana"].apply(lambda x: 1 if (x=="Si")else 0)
########## NUMERICAS ########################
#############################################
#1-Ciclicas (Sen / Cos)
# Transformacion variables ciclicas
#♣Hora
datos['Hora_seno'] = np.sin(2 * np.pi * datos['Hora']/24)
datos['Hora_coseno'] = np.cos(2 * np.pi * datos['Hora']/24)
#Dia
datos['Dia_seno'] = np.sin(2 * np.pi * datos['Dia']/30)
datos['Dia_coseno'] = np.cos(2 * np.pi * datos['Dia']/30)
#Mes
datos['Mes_seno'] = np.sin(2 * np.pi * datos['Mes']/12)
datos['Mes_coseno'] = np.cos(2 * np.pi * datos['Mes']/12)

# Visualizar dia
plt.scatter(datos["Dia"], datos["Dia_seno"])
plt.scatter(datos["Dia"], datos["Dia_coseno"])
# Axis labels
plt.ylabel('Seno y Coseno de Dia')
plt.xlabel('Mes')
plt.title('Seno y Coseno Transformacion (Dia)')

fig, ax = plt.subplots(figsize=(7, 5))
sp = ax.scatter(datos["Dia_seno"], datos["Dia_coseno"], c=datos["Dia"])
ax.set(
    xlabel="Dia_seno",
    ylabel="Dia_coseno",
)
_ = fig.colorbar(sp)

#2- Restantes StandardScaler ( se realizara despues particion datos en entrenamiento/test)
#Las restantes se factoriza
datos['Primary Type'] = pd.factorize(datos["Primary Type"])[0] 
datos['Domestic'] = pd.factorize(datos["Domestic"])[0] 
datos['Arrest'] = pd.factorize(datos["Arrest"])[0] 
datos=datos.drop(["Fecha"],axis=1)
datos=datos.drop(datos.columns[0], axis=1)
datos.to_csv("D:/MASTER/Chicago/datos_num2.csv")
