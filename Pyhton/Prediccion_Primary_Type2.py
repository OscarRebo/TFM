# Importar modulos
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
datos = pd.read_csv('D:/MASTER/Chicago/datos_num2.csv')
datos=datos.drop(datos.columns[0], axis=1)
# datos=datos.drop(["Fecha"],axis=1)
#Calcula la diferencia entre usar StandarScaler y no
X_train, X_test, y_train, y_test = train_test_split(datos.drop('Primary Type', axis=1),
                                                    datos['Primary Type'],
                                                    test_size=0.2,
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

# RED NEURONAL SIN STANDARDSCALER
print("RED NEURONAL SIN STANDARDSCALER")
np.random.seed(42)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1], ) ) )
model.add(Dense(128, activation='relu'))
model.add(Dense(512, activation='relu') )
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=150)
print("Evaluating on Test Dataset")
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# RED NEURONAL CON STANDARDSCALER
print("RED NEURONAL CON STANDARDSCALER")
np.random.seed(42)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1], ) ) )
model.add(Dense(128, activation='relu'))
model.add(Dense(512, activation='relu') )
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=10, batch_size=150)
print("Evaluating on Test Dataset")
scores = model.evaluate(X_test_scaled, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#LOGISTICO CON STANDARSCALER Y SIN EL
lr = LogisticRegression()
lr_scaled = LogisticRegression()
lr.fit(X_train,y_train)
lr_scaled.fit(X_train_scaled,y_train)
# Entrenamiento
y_pred = lr.predict(X_test)
y_pred_scaled = lr_scaled.predict(X_test_scaled)
sin=accuracy_score(y_test,y_pred)
con=accuracy_score(y_test,y_pred_scaled)
print("Sin Estandarizar",round(sin*100,2), "%")
print("Estandarizado", round(con*100,2), "%")

#RANDOM FOREST
print("RANDOM FOREST SIN STANDARDSCALER")
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
sin=accuracy_score(y_test,y_pred)
print("Sin Estandarizar",round(sin*100,2), "%")
print("RANDOM FOREST CON STANDARDSCALER")
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train_scaled, y_train)
y_pred_scaled = rfc.predict(X_test_scaled)
con=accuracy_score(y_test,y_pred_scaled)
print("Estandarizado", round(con*100,2), "%")