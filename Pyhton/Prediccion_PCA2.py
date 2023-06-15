from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.ensemble import GradientBoostingClassifier
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, f1_score,classification_report
from yellowbrick.classifier import ClassificationReport
import warnings
warnings.filterwarnings('ignore')
# Cargar datos
#================================================================
datos=pd.read_csv("D:/MASTER/Chicago/datos_num2.csv")
datos=datos.drop(datos.columns[0], axis=1)
datos.columns
#datos=datos.loc[0:120000]
caracteristicas=['Arrest', 'Beat', "Domestic",'District', 'Ward', 'Community Area',
       'Year', 'Police Districts', 'Police Beats', 'Hora', 'Dia Semana', 'Mes',
       'Dia', 'Fin Semana', 'Franja Dia', 'Delito Grave', 'latPaso', 'lonPaso',
       'Block_Num', 'Block', 'Block_calle', 'PERCENT OF HOUSING CROWDED',
       'PERCENT HOUSEHOLDS BELOW POVERTY', 'PERCENT AGED 16+ UNEMPLOYED',
       'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA',
       'PERCENT AGED UNDER 18 OR OVER 64', 'PER CAPITA INCOME ',
       'HARDSHIP INDEX', 'Safety Score', 'Environment Score',
       'Instruction Score', 'Average Student Attendance',
       'Rate of Misconducts (per 100 students) ', 'Average Teacher Attendance',
       'Individualized Education Program Compliance Rate ',
       'ISAT Exceeding Math %', 'ISAT Exceeding Reading % ',
       'ISAT Value Add Math', 'ISAT Value Add Read',
       'College Enrollment (number of students) ', 'General Services Route ',
       'Partido', 'Temperatura(C)', 'Festivo',
       'Lugar_APARTMENT', 'Lugar_OTROS', 'Lugar_RESIDENCE', 'Lugar_SIDEWALK',
       'Lugar_STREET', 'Hora_seno', 'Hora_coseno', 'Dia_seno', 'Dia_coseno',
       'Mes_seno', 'Mes_coseno']
X=datos.loc[:,caracteristicas].values
y = datos.loc[:,['Primary Type']].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# del datos
pca = PCA()
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
 
explained_variance = pca.explained_variance_ratio_

plt.rcParams["figure.figsize"] = (12,6)
fig, ax = plt.subplots()
xi = np.arange(1, 56, step=1)
y = np.cumsum(pca.explained_variance_ratio_)
plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')
plt.xlabel('Numero de Componentes / Varianza')
plt.xticks(np.arange(0, 55, step=1)) 
plt.ylabel('Varianza Acumulada (%)')
plt.title('Numero de componentes')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% Umbral de Corte', color = 'red', fontsize=16)
ax.grid(axis='x')
plt.show()

pca=PCA(n_components=32)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
#Funcion
def EvaluarModelo(Nombre,resultado): 
    # Evaluación el Modelo
    ac_sc =( accuracy_score(y_test, resultado)*100)
    rc_sc =( recall_score(y_test, resultado, average="weighted")*100)
    pr_sc = (precision_score(y_test, resultado, average="weighted") *100)
    f1_sc = (f1_score(y_test, resultado, average='micro')*100)
    print("RESULTADOS DE: %s " % Nombre.upper())
    print("Accuracy    : ", (ac_sc))
    print("Recall      : ", (rc_sc))
    print("Precision   : ", (pr_sc))
    print("F1 Score    : ", (f1_sc))

def VisualizarModelo(Nombre):    
    sns.regplot(x=y_test, y=y_pred, scatter_kws={'s': 20}, line_kws={'color': 'red'})
    plt.xlabel('Valores Actuales')
    plt.ylabel('Valores predichos')
    plt.title(''+str(Nombre.upper()))
    plt.show()

# REGRESION LOGISTICA
print("#################################")
Rlog = LogisticRegression(random_state = 0)
Rlog.fit(X_train, y_train)
y_pred = Rlog.predict(X_test)
EvaluarModelo('Regresion Logistica',y_pred)
VisualizarModelo("Regresion Logistica")
# RANDOM FOREST
print("#################################")
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
EvaluarModelo('Ramdom Forest',y_pred)
VisualizarModelo("Ramdom Forest")
# ADABOOST CLASIFICADOR
print("#################################")
# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = model.predict(X_test)
EvaluarModelo('Adaboost',y_pred)
VisualizarModelo("Adaboost")
#GAUSIANO
print("#################################")
gausiano = GaussianNB()
gausiano.fit(X_train, y_train)
# Se aplica 10 pasos de validacion cruzada
gaussian_accuracy = (cross_val_score(gausiano, X_test, y_test, cv=10).mean())*100
gaussian_precision= cross_val_score(gausiano, X_test, y_test, cv=8, scoring='precision').mean()
gaussian_recall = cross_val_score(gausiano, X_test, y_test, cv=8, scoring='recall').mean()
print("RESULTADOS DE GAUSSIANO")
print("Accuracy: ",round(gaussian_accuracy),2)
# print("Recall :",gaussian_recall)
# print("Precision : ",gaussian_precision)
# VisualizarModelo("GAUSSIANNB")

#Gradient
print("#################################")
grad = GradientBoostingClassifier(learning_rate=0.5,n_estimators = 10, random_state = 42)
grad.fit(X_train, y_train)
y_pred = grad.predict(X_test) 
EvaluarModelo('Gradient',y_pred)
VisualizarModelo("GradientBoostingClassifier")
# Combirnar los modelos para crear el ENSEMBLE MODEL
print("########################################################")
print("############### ENSEMBLE VOTING MODEL ##################")
evm_model = VotingClassifier(estimators=[('Rlog', Rlog), ('rfc', rfc), 
                             ('abc', abc), ('gausiano', gausiano), ('grad', grad)], 
                         weights=[1,1,1,1,1],
                         flatten_transform=True)

#Entrenar
evm_model.fit(X=X_train, y=y_train)   
# Predecir
result = evm_model.predict(X_test)
#Evaluar el Modelo
EvaluarModelo('Ensemble Voting Model',result)
# confusion_m = confusion_matrix(y_test, result)
# print(confusion_m)
VisualizarModelo("ENSEMBLE VOTING MODEL")