# Importar modulos
import pandas as pd 
import numpy as np 
import seaborn as sns
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
datos = pd.read_csv('D:/MASTER/Chicago/datos_num2.csv')
datos=datos.drop(datos.columns[0], axis=1)
# datos=datos.loc[0:1000]
inicio=time.time()
X = datos.drop('Primary Type', axis=1)
y = datos['Primary Type']
# division
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
X_train.shape, X_test.shape
# aplicar StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("############## MODELO KNN ###############")
# best_k=23
print("Bucando mejor k :")
#Busqueda mejor "k"
k_values = [i for i in range (1,25)]
scores = []
scaler = StandardScaler()
X = scaler.fit_transform(X)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=5)
    scores.append(np.mean(score))
    print("k=",k, "y validacion cruzada= ",(np.mean(score)))
sns.lineplot(x = k_values, y = scores, marker = 'o')
plt.xlabel("Valores de K")
plt.ylabel("Precision")
best_index = np.argmax(scores)
best_k = k_values[best_index]
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Precisicon con k= ",best_k, "es de :", (accuracy*100))
fin=time.time()
print("Tiempo Total")
str(datetime.timedelta(seconds=(fin-inicio)))
print(fin-inicio)