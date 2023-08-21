# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# Dfprocess = pd.read_csv(r"C:\Users\Flavio Ruvalcaba\Documents\Escuela\Universidad\7_Semestre\InteligenciaArtificialGpo101\inputData\test.csv")
# print(Dfprocess.head())



# def hypotesis(a, b):
#     acum = 0
#     for i in():
# 		acum += 
# 	return acum;

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paso 1: Cargar los datos
data_path = r"C:\Users\Flavio Ruvalcaba\Documents\Escuela\Universidad\7_Semestre\InteligenciaArtificialGpo101\inputData\Housing.csv"
data = pd.read_csv(data_path)
y = data['price']
x = data['bathrooms']

# Paso 2: Inicializar parámetros
learning_rate = 0.001
iterations = 1000
n = len(x)
slope = 0
intercept = 0

# Paso 3: Implementar la regresión lineal usando Gradient Descent
for _ in range(iterations):
    # Calcular las predicciones (hipótesis)
    predictions = slope * x + intercept
    
    # Calcular el error
    errors = predictions - y
    
    # Calcular las derivadas parciales para el gradiente
    slope_gradient = (2/n) * np.sum(errors * x)
    intercept_gradient = (2/n) * np.sum(errors)
    
    # Actualizar los coeficientes utilizando el gradiente descendente
    slope -= learning_rate * slope_gradient
    intercept -= learning_rate * intercept_gradient

# Paso 4: Hacer predicciones con el modelo entrenado
new_x = 10
predicted_y = slope * new_x + intercept
print("Predicted y:", predicted_y)

# Paso 5: Visualizar los resultados
plt.scatter(x, y, label='Data')
plt.plot(x, slope * x + intercept, color='red', label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()