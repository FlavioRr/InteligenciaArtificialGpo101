

# ###################REGRESION LOGISTICA###################
#
# En este proyecto implementaremos una regresion logistica con el uso de un 
# framework
# Con un set de datos obtenidos de Kaggle, vamos a predecir si los estudiantes aprobaron
# o reprobaron segun sus horas de estudio y gpa.



# importamos las librarias necesarias para ejecutar el modelo
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, r2_score
from sklearn.model_selection import cross_val_score

# Cargamos los datos
# En mi variable data_path  anexo el path donde se encuentran los datos que quiero utilizar.
data_path_train = "inputData/gpa_study_hours_train.csv"
data_path_test = "inputData/gpa_study_hours_test.csv"
data_path_validation = "inputData/gpa_study_hours_validation.csv"
#Con la libreria pandas, utilizo el metodo read_csv para poder acceder al documento csv.
data_train = pd.read_csv(data_path_train)
data_test = pd.read_csv(data_path_test)
data_validation = pd.read_csv(data_path_validation)
#Creo dataframes para manipular las columnas del documento csv.
df_train = pd.DataFrame(data_train, columns=["gpa", "study_hours", "graduated"])
df_test = pd.DataFrame(data_test, columns=["gpa", "study_hours", "graduated"])
df_validation = pd.DataFrame(data_validation, columns=["gpa", "study_hours", "graduated"])

# En la variable x identifico los valores de la tabla de datos que se obtienen de una observacion. En
# este caso son gpa y study_hours.
# En la variable y identifico los valores de la tabla de datos que existen como clasificacion binaria 
# si un estudiante aprobo o reprobo. En este caso es la columna graduated

x_train = df_train[["gpa", "study_hours"]]
y_train = df_train["graduated"]
x_test = df_test[["gpa", "study_hours"]]
y_test = df_test["graduated"]
x_validation = df_validation[["gpa", "study_hours"]]
y_validation = df_validation["graduated"]

# Iniciamos a entrenar el modelo 

#Utilizamos la libreria sklearn utilizando regrsion logistica.
model = LogisticRegression(penalty=None, max_iter=1000)
model.fit(x_train, y_train)

#Empezamos a crear predicciones con nuestros datos de test.
y_pred = model.predict(x_test)
y_pred_validation = model.predict(x_validation)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_val = confusion_matrix(y_validation, y_pred_validation)

# Calcular True Positives (TP), True Negatives (TN), False Positives (FP) y False Negatives (FN)
tn, fp, fn, tp = conf_matrix.ravel()
tnv, fpv, fnv, tpv = conf_matrix_val.ravel()

# Calcular Precisión (Precision)
precision = precision_score(y_test, y_pred)

# Calcular Exactitud (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
accuracy_validation = accuracy_score(y_validation, y_pred_validation)

#Recall
recall = tp/(tp+fn)

# También puedes evaluar el modelo en el conjunto de validación y test
r2_validation = r2_score(y_validation, y_pred_validation)
r2_test = r2_score(y_test, y_pred)


# Imprimir los resultados
print(f'test True Positives (TP): {tp}')
print(f'test True Negatives (TN): {tn}')
print(f'test False Positives (FP): {fp}')
print(f'test False Negatives (FN): {fn}')
print(f'validation True Positives (TP): {tpv}')
print(f'validation True Negatives (TN): {tnv}')
print(f'validation False Positives (FP): {fpv}')
print(f'validation False Negatives (FN): {fnv}')
print(f'Recall: {recall}')
print(f'Precision: {precision * 100:.2f}%')
print(f'Accuracy en test: {accuracy * 100:.2f}%')
print(f'R² ajustado en test: {r2_test:.2f}')
print(f'Accuracy en validation: {accuracy_validation * 100:.2f}%')
print(f'R² ajustado en validation: {r2_validation:.2f}')




# Realizar cross-validation con k-fold 
k = 4
scores = cross_val_score(model, x_train, y_train, cv=k, scoring='accuracy')

# Calcula la precisión promedio y su desviación estándar
mean_accuracy = np.mean(scores)
std_accuracy = np.std(scores)

print(f'Precisión promedio en cross-validation ({k}-fold): {mean_accuracy:.2f} +/- {std_accuracy:.2f}')







