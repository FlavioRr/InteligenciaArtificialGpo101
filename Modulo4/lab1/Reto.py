import json

# Cargar el dataset desde el archivo JSON
with open(r"C:\Users\Flavio Ruvalcaba\Documents\Escuela\Universidad\7_Semestre\InteligenciaArtificialGpo101\Modulo4\lab1\iris.json") as json_file:
    dataset = json.load(json_file)

# Imprimir el primer elemento del dataset para verificar la carga
print(dataset[0])

from flask import Flask, request, jsonify

app = Flask(__name__)

# Ruta para obtener información filtrada por edad
@app.route('/filtrar-por-tamaño', methods=['GET'])
def filtrar_por_tamaño():
    tamaño_min = int(request.args.get('min'))
    tamaño_max = int(request.args.get('max'))

    # Filtrar el dataset por edad
    resultado = [iris for iris in dataset if tamaño_min <= iris['sepalLength'] <= tamaño_max]

    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True)

