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


# Paso 1: Cargar los datos
#En mi variable data_path  anexo el path donde se encuentran los datos que quiero utilizar.
data_path = "inputData/gpa_study_hours.csv"
#Con la libreria pandas, utilizo el metodo read_csv para poder acceder al documento csv.
data = pd.read_csv(data_path)
# En la variable x e y identifico las columnas de la tabla de datos que quiero buscar su relacion.
y = data['study_hours']
x = data['gpa']