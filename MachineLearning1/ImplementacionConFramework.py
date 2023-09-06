# ###################REGRESION LOGISTICA###################
 #
# En este proyecto implementaremos una regresion logistica con el uso de un 
# framework

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score




# Iniciamos a entrenar el modelo 
# Cargamos los datos
# En mi variable data_path  anexo el path donde se encuentran los datos que quiero utilizar.
data_path_train = "inputData/gpa_study_hours_train.csv"
data_path_test = "inputData/gpa_study_hours_test.csv"
#Con la libreria pandas, utilizo el metodo read_csv para poder acceder al documento csv.
data_train = pd.read_csv(data_path_train)
data_test = pd.read_csv(data_path_test)
#Creo dataframes para manipular las columnas del documento csv.
df_train = pd.DataFrame(data_train, columns=["gpa", "study_hours", "graduated"])
df_test = pd.DataFrame(data_test, columns=["gpa", "study_hours", "graduated"])

# En la variable x identifico los valores de la tabla de datos que se obtienen de una observacion. En
# este caso son gpa y study_hours.
# En la variable y identifico los valores de la tabla de datos que existen como clasificacion binaria 
# si un estudiante aprobo o reprobo. En este caso es la columna graduated

x_train = df_train[["gpa", "study_hours"]]
y_train = df_train["graduated"]
x_test = df_test[["gpa", "study_hours"]]
y_test = df_test["graduated"]





#Definimos la cantidad de iteraciones que se utilizaran en el modelo.

#Utilizamos la libreria sklearn utilizando regrsion logistica.
model = LogisticRegression(penalty='none', max_iter=1000)
model.fit(x_train, y_train)

#Empezamos a crear predicciones con nuestros datos de test.
y_pred = model.predict(x_test)
print(y_pred)
#Calculamos la precision de la prediccion con accuracy_score.
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')