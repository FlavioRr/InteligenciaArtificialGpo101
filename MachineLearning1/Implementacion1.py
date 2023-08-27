
###################REGRESION LINEAL###################
#
# Cuantas horas tengo que estudiar para obtener un GPA de 4?
#
# Con un set de datos obtenidos de Kaggle, vamos a predecir la calificacion de los estudiantes
# segun la cantidad de horas que estudiaron


#importo la libreria numpy para facilitar el uso de vectores y matrices.
import numpy as np
#importo la libreria de pandas para la manipulación y el análisis de datos.
import pandas as pd
#importo la libreria de matplotlib para generar graficos en dos dimensiones.
import matplotlib.pyplot as plt

# Paso 1: Cargar los datos
#En mi variable data_path  anexo el path donde se encuentran los datos que quiero utilizar.
data_path = r"C:\Users\Flavio Ruvalcaba\Documents\Escuela\Universidad\7_Semestre\InteligenciaArtificialGpo101\inputData\gpa_study_hours.csv"
#Con la libreria pandas, utilizo el metodo read_csv para poder acceder al documento csv.
data = pd.read_csv(data_path)
# En la variable x e y identifico las columnas de la tabla de datos que quiero buscar su relacion.
y = data['study_hours']
x = data['gpa']

# Paso 2: Inicializar parámetros
#Learning rate es el parametro (modificable) que se utiliza en el proposito de optimizacion.
#Learning rate es la magnitud de "pasos" que hara el algoritmo para encontrar la solucion de la
#manera mas rapida y eficiente, es muy importante modificarlo para observar como el algoritmo se adapta a el.
#En caso de que este sea muy grande su precision podria verse afectada, sin embargo, si es demasiado pequeña 
#la optimizacion seria increiblemente lenta, es necesario encontrar un balance entre el "paso" de aprendizaje.
learning_rate = 0.001
#Este es el numero de iteraciones que se implementaran en el ciclo de aprendizaje 
iterations = 1000
# n es el numero de valores que tiene los datos que se van a utilizar en esta ocasion, calificaciones de gpa
n = len(x)

# θ o theta 1 representa en la hipotesis el cambio en la pendiente cada que x aumenta 
θ = 0
# b, bias, intercept o theta 0 es la representacion en la hipotesis del valor de y cuando x esta en 0
b = 0

# Paso 3: Implementar la regresion lineal usando gradiente descendiente
for _ in range(iterations):
    # Se define la hipotesis (hypotesis = θx + b)
    h = θ * x + b
    # Calculamos el costo (que tan equivocados estamos con nuestra hipotesis(h) con el valor real(y))
    errors = h - y
    #los valores b_gradient y θ_gradient calculan cuanto tienen que modificarse θ y b para acercarse 
	# mas a la respuesta correcta
    θ_gradient = (2/n) * np.sum(errors * x)
    b_gradient = (2/n) * np.sum(errors)
    
    # Actualizar los coeficientes utilizando el gradiente descendente
    θ -= learning_rate * θ_gradient
    b -= learning_rate * b_gradient
    
    print("Hipotesis = ", h)
    print("Theta1 = ", θ)
    print("Theta0 = ", b)

# Paso 4: Hacer predicciones con el modelo entrenado
#ingresamos un nuevo valor en x que equivale un valor de GPA, cuanto tengo que estudiar para obtener esta calificacion?
nuevo_x = 4
#una vez que entrenamos nuestros theta 0 y theta 1, predecimos el valor de y
predicted_y = θ * nuevo_x + b
print("Predicted y:", predicted_y)

# Paso 5: Visualizar los resultados
# con la libreria matplotlib plotearemos los datos de kaggle y la linea regresiva que hemos creado
plt.scatter(x, y, label='Data')
plt.plot(x, θ * x + b, color='red', label='Regresion Lineal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()