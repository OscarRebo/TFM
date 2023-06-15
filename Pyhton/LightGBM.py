# Importar modulos
import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
datos = pd.read_csv('D:/MASTER/Chicago/datos_num2.csv')
datos=datos.drop(datos.columns[0], axis=1)
X_train, X_test, y_train, y_test = train_test_split(datos.drop('Primary Type', axis=1),
                                                    datos['Primary Type'],
                                                    test_size=0.3,
                                                    random_state=0)
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print('LightGBM Model precision:',round(accuracy*100,2), "%")
print('Precision Training: {:.4f}'.format(clf.score(X_train, y_train)))
print('Precision Test: {:.4f}'.format(clf.score(X_test, y_test)))
# Visualizar los 100 primeros resultados
y_test2=y_test[0:100]
y_pred2=y_pred[0:100]
x_ax = range(len(y_test2))
plt.figure(figsize=(15, 6))
plt.plot(x_ax, y_test2, label="Original")
plt.plot(x_ax, y_pred2, label="Predicho")
plt.title("Prediccion de los 100 primeros resultados (LightBGM)")
plt.xlabel('Numero')
plt.ylabel('Primary Type')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show() 