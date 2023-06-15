# Importar modulos
import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.layers import Dense, Dropout
from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')
#Cargar datos
datos=pd.read_csv("D:/MASTER/Chicago/datos2.csv")
datos=datos.drop(datos.columns[0], axis=1)
# datos=datos.drop("Date", axis=1)
datos.columns

agrupar = pd.pivot_table(datos[["Fecha", "Primary Type", "Community Area","Arrest", "Hora"
                                ]], index=datos[["Fecha", "Community Area","Primary Type"]], aggfunc="count")
agrupar.reset_index(inplace=True)
agrupar.rename(columns={"Arrest":"Delitos Totales"}, inplace=True)
agrupar.head(10)

agrupar["Año"] = pd.DatetimeIndex(agrupar["Fecha"]).year
agrupar["Mes"] = pd.DatetimeIndex(agrupar["Fecha"]).month
agrupar["Dia"] = pd.DatetimeIndex(agrupar["Fecha"]).day
agrupar["Dia Semana"] = pd.DatetimeIndex(agrupar["Fecha"]).dayofweek

#♣Hora
agrupar['Mes_seno'] = np.sin(2 * np.pi * agrupar['Mes']/30)
agrupar['Mes_coseno'] = np.cos(2 * np.pi * agrupar['Mes']/30)
#Dia
agrupar['Dia_seno'] = np.sin(2 * np.pi * agrupar['Dia']/30)
agrupar['Dia_coseno'] = np.cos(2 * np.pi * agrupar['Dia']/30)

agrupar['Dia_semana_seno'] = np.sin(2 * np.pi * agrupar['Dia Semana']/7)
agrupar['Dia_semana_coseno'] = np.cos(2 * np.pi * agrupar['Dia Semana']/7)

#♣Hora
agrupar['Hora_seno'] = np.sin(2 * np.pi * agrupar['Hora']/24)
agrupar['Hora_coseno'] = np.cos(2 * np.pi * agrupar['Hora']/24)

agrupar=pd.get_dummies(agrupar,columns=["Año"])

agrupar=agrupar.drop(["Mes","Dia","Dia Semana","Hora"], axis=1)

# variables=variables.loc[0:1000000]
# agrupar['Primary Type'] = pd.factorize(agrupar["Primary Type"])[0] 
agrupar=agrupar.drop(["Fecha","Primary Type"],axis=1)
X_train, X_test, y_train, y_test = train_test_split(agrupar.drop('Delitos Totales', axis=1),
                                                    agrupar['Delitos Totales'],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
scaler = StandardScaler()
scaler.fit(X_train)
# Se transforma con StandardScaler por separado
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Se transforma a DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
X_train=X_train_scaled
X_test=X_test_scaled
del X_train_scaled,X_test_scaled
#REGRESION LOGISTICA
print("REGRESION LOGISTICA")
lr = LogisticRegression()
lr_scaled = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print("Precision",accuracy_score(y_test,y_pred))

#RANDOM FOREST
from sklearn.linear_model import LogisticRegression
print("RANDOM FOREST")
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
sin=accuracy_score(y_test,y_pred)
print("Sin Estandarizar",round(sin*100,2), "%")

# RED NEURONAL
print("RED NEURONAL")
np.random.seed(42)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1], ) ) )
model.add(Dense(128, activation='relu'))
model.add(Dense(512, activation='relu') )
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=150)
print("Evaluating on Test Dataset")
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))