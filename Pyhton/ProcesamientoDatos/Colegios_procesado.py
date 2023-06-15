# Importar modulos
import pandas as pd 
#Datos colegios
colegios1=pd.read_csv('D:/MASTER/Chicago/Chicago_Public_Schools_-_Progress_Report_Cards__2011-2012_.csv')
#Se renombra el campo Community area de la base socioeconomico1 para que coincida en todas las bases
colegios = colegios1.rename(columns={'Community Area Number': 'Community Area'})
print ("Tipos de datos")
print ("**************")
colegios.info()
colegios.columns
# ANALISIS DE VARIABLES
#Se eliminan campos que no aportan valor al modelo
colegios=colegios.drop(["School ID","Name of School","Street Address","City","State",
                        "ZIP Code","Phone Number","Link ","Network Manager","Healthy Schools Certified?",
                        "Collaborative Name","Adequate Yearly Progress Made? ","Instruction Icon ",
                        "Leaders Icon ","Safety Icon ","Teachers Icon ","Parent Engagement Icon ",
                        "Parent Environment Icon","ISAT Value Add Color Math","ISAT Value Add Color Read",
                        "Freshman on Track Rate %","RCDTS Code","X_COORDINATE","Y_COORDINATE",
                        "Latitude","Longitude","Community Area Name","Ward","Police District",
                        "Location"],axis=1)
# #2 Tratamiento de datos
#Se calcula la media por Area Comunitaria
agrupado = colegios.groupby(["Community Area"]).mean()
agrupado=round(agrupado,2)
numericas=colegios._get_numeric_data().columns
print("Las variables numericas son :",list(numericas))
print("Las variables categoricas son :",list(set(colegios.columns)-set(numericas)))

#1- Existen valores faltantes
agrupado.isnull().sum().reset_index()
#Se decide rellenar esos valores nulos con las medias del anterior y posterior
#asi no se pierde esa Area Comunitaria
valor=[]
valor=agrupado.iloc[31]

anterior=agrupado.iloc[30]["ISAT Exceeding Math %"]
posterior=agrupado.iloc[32]["ISAT Exceeding Math %"]
valor["ISAT Exceeding Math %"]=((posterior+anterior)/2)

anterior=agrupado.iloc[30]["ISAT Exceeding Reading % "]
posterior=agrupado.iloc[32]["ISAT Exceeding Reading % "]
valor["ISAT Exceeding Reading % "]=((posterior+anterior)/2)


anterior=agrupado.iloc[30]["ISAT Value Add Math"]
posterior=agrupado.iloc[32]["ISAT Value Add Math"]
valor["ISAT Value Add Math"]=((posterior+anterior)/2)

anterior=agrupado.iloc[30]["ISAT Value Add Read"]
posterior=agrupado.iloc[32]["ISAT Value Add Read"]
valor["ISAT Value Add Read"]=((posterior+anterior)/2)
agrupado.iloc[31]=valor

valor=agrupado.iloc[35]
anterior=agrupado.iloc[34]["Safety Score"]
posterior=agrupado.iloc[36]["Safety Score"]
valor["Safety Score"]=((posterior+anterior)/2)

anterior=agrupado.iloc[34]["Environment Score"]
posterior=agrupado.iloc[36]["Environment Score"]
valor["Environment Score"]=((posterior+anterior)/2)

anterior=agrupado.iloc[34]["Instruction Score"]
posterior=agrupado.iloc[36]["Instruction Score"]
valor["Instruction Score"]=((posterior+anterior)/2)


anterior=agrupado.iloc[34]["ISAT Value Add Math"]
posterior=agrupado.iloc[36]["ISAT Value Add Math"]
valor["ISAT Value Add Math"]=((posterior+anterior)/2)

anterior=agrupado.iloc[34]["ISAT Value Add Read"]
posterior=agrupado.iloc[36]["ISAT Value Add Read"]
valor["ISAT Value Add Read"]=((posterior+anterior)/2)
agrupado.iloc[35]=valor

agrupado.head(5)

agrupado.to_csv("D:/MASTER/Chicago/colegios.csv")