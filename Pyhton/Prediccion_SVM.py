# Importar modulos
import pandas as pd 
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn import svm
import warnings
warnings.filterwarnings('ignore')
datos = pd.read_csv('D:/MASTER/Chicago/datos_num.csv')
datos=datos.drop(datos.columns[0], axis=1)
datos=datos.loc[0:100000]
inicio=time.time()
X = datos.drop('Primary Type', axis=1)
y = datos['Primary Type']
# division
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
X_train.shape, X_test.shape

# Aplicar StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("############## MODELO SVM ###############")
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Precision : {0:0.4f}'. format(accuracy_score(y_test, y_pred)*100))
print("-------------Tiempos----------------")
fin=time.time()
print("Tiempo Total")
str(datetime.timedelta(seconds=(fin-inicio)))
print(fin-inicio)