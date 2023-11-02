import os
import shutil
import random

def split_data(input_dir, output_train_dir, output_test_dir, output_validation_dir, train_split, test_split):
    """
    Divide un conjunto de datos en entrenamiento, prueba y validación manteniendo la estructura de carpetas.
    
    Args:
        input_dir (str): Directorio principal con carpetas secundarias "man" y "woman".
        output_train_dir (str): Directorio de destino para los datos de entrenamiento.
        output_test_dir (str): Directorio de destino para los datos de prueba.
        output_validation_dir (str): Directorio de destino para los datos de validación.
        train_split (float): Porcentaje de datos para entrenamiento.
        test_split (float): Porcentaje de datos para prueba.
    """
    # Recorre las carpetas "man" y "woman"
    for gender in ["man", "woman"]:
        source_dir = os.path.join(input_dir, gender)
        all_files = os.listdir(source_dir)
        random.shuffle(all_files)  # Mezclar los archivos aleatoriamente

        num_files = len(all_files)
        num_train = int(train_split * num_files)
        num_test = int(test_split * num_files)

        train_files = all_files[:num_train]
        test_files = all_files[num_train:num_train + num_test]
        validation_files = all_files[num_train + num_test:]

        # Crea las carpetas de destino
        os.makedirs(os.path.join(output_train_dir, gender), exist_ok=True)
        os.makedirs(os.path.join(output_test_dir, gender), exist_ok=True)
        os.makedirs(os.path.join(output_validation_dir, gender), exist_ok=True)

        # Copia las imágenes a las carpetas correspondientes
        for file in train_files:
            src = os.path.join(source_dir, file)
            dst = os.path.join(output_train_dir, gender, file)
            shutil.copy(src, dst)

        for file in test_files:
            src = os.path.join(source_dir, file)
            dst = os.path.join(output_test_dir, gender, file)
            shutil.copy(src, dst)

        for file in validation_files:
            src = os.path.join(source_dir, file)
            dst = os.path.join(output_validation_dir, gender, file)
            shutil.copy(src, dst)

# Ejemplo de uso
input_dir = r"C:\Users\Flavio Ruvalcaba\Downloads\archive (7)\faces"
output_train_dir = r"C:\Users\Flavio Ruvalcaba\Documents\Escuela\Universidad\7_Semestre\inputData\faces\train"
output_test_dir = r"C:\Users\Flavio Ruvalcaba\Documents\Escuela\Universidad\7_Semestre\inputData\faces\test"
output_validation_dir = r"C:\Users\Flavio Ruvalcaba\Documents\Escuela\Universidad\7_Semestre\inputData\faces\validation"
train_split = 0.7
test_split = 0.2

split_data(input_dir, output_train_dir, output_test_dir, output_validation_dir, train_split, test_split)
