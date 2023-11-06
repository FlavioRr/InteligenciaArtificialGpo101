import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Obtener la ruta completa del directorio del script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta completa al archivo de imagen en el mismo directorio que el script
image_name = "man_9762.jpg"
image_path = os.path.join(script_directory, image_name)

# Cargar el modelo previamente guardado
model_path = "vgg_man_woman_model.h5"  
model_path = os.path.join(script_directory, model_path)
model = load_model(model_path)

# Ahora puedes utilizar el modelo para hacer predicciones en la imagen
img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
img = tf.keras.preprocessing.image.img_to_array(img)
img = img / 255.0  # Normaliza la imagen
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

# Realizar la predicción
predictions = model.predict(img)

# El resultado será una probabilidad, puedes interpretarlo según tu problema
if predictions[0] > 0.5:
    print('La imagen representa a una mujer.')
else:
    print('La imagen representa a un hombre.')

